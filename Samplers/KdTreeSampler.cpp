#include "KdTreeSampler.h"

#include <algorithm>
#include <numeric>

#include <CellarWorkbench/Misc/Log.h>
#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/GpuMesh.h"
#include "DataStructures/Tetrahedralizer.h"

#include "LocalSampler.h"

using namespace cellar;


struct KdNode
{
    KdNode() :
        left(nullptr),
        right(nullptr)
    {}

    ~KdNode()
    {
        delete left;
        delete right;
    }

    KdNode* left;
    KdNode* right;

    glm::dvec4 separator;
    glm::dvec3 minBox;
    glm::dvec3 maxBox;

    MeshMetric metric;
};


// CUDA Drivers Interface
void installCudaKdTreeSampler();
void updateCudaKdNodes(
        const std::vector<GpuKdNode>& kdNodesBuff);

KdTreeSampler::KdTreeSampler() :
    AbstractSampler("Kd-Tree", ":/glsl/compute/Sampling/KdTree.glsl", installCudaKdTreeSampler),
    _debugMesh(new Mesh()),
    _kdNodesSsbo(0)
{
}

KdTreeSampler::~KdTreeSampler()
{
    glDeleteBuffers(1, &_kdNodesSsbo);
    _kdNodesSsbo = 0;
}

bool KdTreeSampler::isMetricWise() const
{
    return true;
}

void KdTreeSampler::updateGlslData(const Mesh& mesh) const
{
    if(_kdNodesSsbo == 0)
        glGenBuffers(1, &_kdNodesSsbo);

    GLuint kdNodes    = mesh.glBufferBinding(EBufferBinding::KD_NODES_BUFFER_BINDING);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kdNodes,    _kdNodesSsbo);


    // Build GPU Buffers
    {
        std::vector<GpuKdNode> gpuKdNodes;
        buildGpuBuffers(_rootNode.get(), gpuKdNodes);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _kdNodesSsbo);
        size_t kdNodesSize = sizeof(decltype(gpuKdNodes.front())) * gpuKdNodes.size();
        glBufferData(GL_SHADER_STORAGE_BUFFER, kdNodesSize, gpuKdNodes.data(), GL_STREAM_COPY);
    }
}

void KdTreeSampler::updateCudaData(const Mesh& mesh) const
{
    // Build GPU Buffers
    {
        std::vector<GpuKdNode> gpuKdNodes;
        buildGpuBuffers(_rootNode.get(), gpuKdNodes);

        updateCudaKdNodes(gpuKdNodes);
    }
}

void KdTreeSampler::clearGlslMemory(const Mesh& mesh) const
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _kdNodesSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STREAM_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void KdTreeSampler::clearCudaMemory(const Mesh& mesh) const
{
    {
        std::vector<GpuKdNode> gpuKdNodes;
        updateCudaKdNodes(gpuKdNodes);
    }
}

void KdTreeSampler::setReferenceMesh(
        const Mesh& mesh)
{
    size_t vertCount = mesh.verts.size();

    // Clear resources
    _debugMesh.reset();
    _rootNode.reset();


    if(vertCount == 0)
    {
        _rootNode.reset(new KdNode());
        _rootNode->metric = MeshMetric(1.0);

        getLog().postMessage(new Message('I', false,
            "Creating single cell for empty mesh",
            "KdTreeSampler"));

        return;
    }
    else
    {
        // Compute Kd Tree depth
        int height = discretizationDepth();
        if(height < 0)
        {
            height = (int)std::log2(std::ceil(vertCount/3.0));
        }

        getLog().postMessage(new Message('I', false,
            "Sampling mesh metric in a Kd-Tree",
            "KdTreeSampler"));
        getLog().postMessage(new Message('I', false,
            "Maximum Kd-Tree's depth: " + std::to_string(height),
            "KdTreeSampler"));

        std::vector<uint> xSort(vertCount);
        std::iota(xSort.begin(), xSort.end(), 0);
        std::vector<uint> ySort(xSort);
        std::vector<uint> zSort(xSort);

        std::sort(xSort.begin(), xSort.end(), [&mesh](uint a, uint b){
            return mesh.verts[a].p.x < mesh.verts[b].p.x;});
        std::sort(ySort.begin(), ySort.end(), [&mesh](uint a, uint b){
            return mesh.verts[a].p.y < mesh.verts[b].p.y;});
        std::sort(zSort.begin(), zSort.end(), [&mesh](uint a, uint b){
            return mesh.verts[a].p.z < mesh.verts[b].p.z;});


        LocalSampler localSampler;
        localSampler.setScaling(scaling());
        localSampler.setAspectRatio(aspectRatio());
        localSampler.setReferenceMesh(mesh);

        // Fill Sampler's data strucutres
        glm::dvec3 minBounds, maxBounds;
        boundingBox(mesh, minBounds, maxBounds);

        _rootNode.reset(new KdNode());
        build(_rootNode.get(), height,
              mesh, localSampler,
              minBounds, maxBounds,
              xSort, ySort, zSort);
    }
}

MeshMetric KdTreeSampler::metricAt(
        const glm::dvec3& position,
        uint& cachedRefTet) const
{
    const MeshMetric METRIC_ERROR(0.0);

    KdNode* node = nullptr;
    KdNode* child = _rootNode.get();

    while(child != nullptr)
    {
        node = child;

        double dist = child->separator.w;
        glm::dvec3 axis(child->separator);
        if(glm::dot(position, axis) - dist < 0.0)
            child = node->left;
        else
            child = node->right;
    }

    if(node != nullptr)
    {
        return node->metric;
    }
    else
    {
        return METRIC_ERROR;
    }
}

void KdTreeSampler::releaseDebugMesh()
{
    _debugMesh.reset();
}

const Mesh& KdTreeSampler::debugMesh()
{
    if(_debugMesh.get() == nullptr)
    {
        _debugMesh.reset(new Mesh());

        if(_rootNode.get() != nullptr)
        {
            meshTree(_rootNode.get(), *_debugMesh);

            _debugMesh->modelName = "Kd-Tree Sampling Mesh";
            _debugMesh->compileTopology();
        }
    }

    return *_debugMesh;
}

void KdTreeSampler::build(
        KdNode* node,
        int height,
        const Mesh& mesh,
        const AbstractSampler& localSampler,
        const glm::dvec3& minBox,
        const glm::dvec3& maxBox,
        std::vector<uint>& xSort,
        std::vector<uint>& ySort,
        std::vector<uint>& zSort)
{
    node->minBox = minBox;
    node->maxBox = maxBox;

    size_t vertCount = xSort.size();
    if(height > 0 && vertCount > 3)
    {
        glm::dvec3 extents(
            mesh.verts[xSort.back()].p.x - mesh.verts[xSort.front()].p.x,
            mesh.verts[ySort.back()].p.y - mesh.verts[ySort.front()].p.y,
            mesh.verts[zSort.back()].p.z - mesh.verts[zSort.front()].p.z);

        // Find separator position and axis
        glm::dvec3 axis = glm::dvec3(1, 0, 0);
        size_t sepIdR = vertCount/2;
        size_t sepIdL = sepIdR - 1;

        uint sepL = xSort[sepIdL];
        uint sepR = xSort[sepIdR];
        if(extents.y >= extents.x && extents.y >= extents.z)
        {
            axis = glm::dvec3(0, 1, 0);
            sepL = ySort[sepIdL];
            sepR = ySort[sepIdR];
        }
        else if(extents.z >= extents.x && extents.z >= extents.y)
        {
            axis = glm::dvec3(0, 0, 1);
            sepL = zSort[sepIdL];
            sepR = zSort[sepIdR];
        }

        glm::dvec3 sepPos;
        if(vertCount % 2 == 0)
            sepPos = (mesh.verts[sepL].p + mesh.verts[sepR].p) / 2.0;
        else
            sepPos = mesh.verts[sepR].p;

        double sepVal = glm::dot(axis, sepPos);
        node->separator = glm::dvec4(axis, sepVal);


        // Distribute vertices around the separator
        std::vector<uint> xSortL;
        std::vector<uint> xSortR;
        std::vector<uint> ySortL;
        std::vector<uint> ySortR;
        std::vector<uint> zSortL;
        std::vector<uint> zSortR;
        for(size_t v=0; v < vertCount; ++v)
        {
            double xDist = glm::dot(mesh.verts[xSort[v]].p, axis) - sepVal;
            if(xDist <= 0.0)
                xSortL.push_back(xSort[v]);
            if(xDist >= 0.0)
                xSortR.push_back(xSort[v]);

            double yDist = glm::dot(mesh.verts[ySort[v]].p, axis) - sepVal;
            if(yDist <= 0.0)
                ySortL.push_back(ySort[v]);
            if(yDist >= 0.0)
                ySortR.push_back(ySort[v]);

            double zDist = glm::dot(mesh.verts[zSort[v]].p, axis) - sepVal;
            if(zDist <= 0.0)
                zSortL.push_back(zSort[v]);
            if(zDist >= 0.0)
                zSortR.push_back(zSort[v]);
        }


        // Deallocate sorted vertices vectors
        xSort.clear();
        xSort.shrink_to_fit();
        ySort.clear();
        ySort.shrink_to_fit();
        zSort.clear();
        zSort.shrink_to_fit();


        // Child bounding boxes
        glm::dvec3 minBoxL = minBox;
        glm::dvec3 maxBoxL = glm::mix(maxBox, sepPos, axis);

        glm::dvec3 minBoxR = glm::mix(minBox, sepPos, axis);
        glm::dvec3 maxBoxR = maxBox;

        // Build children nodes
        node->left = new KdNode();
        build(node->left,  height-1, mesh, localSampler, minBoxL, maxBoxL, xSortL, ySortL, zSortL);

        node->right = new KdNode();
        build(node->right, height-1, mesh, localSampler, minBoxR, maxBoxR, xSortR, ySortR, zSortR);
    }
    else
    {
        assert(vertCount > 0);
        node->separator.w = 0.0;

        glm::dvec3 meanPos;
        for(uint v : xSort)
        {
            meanPos += mesh.verts[v].p;
        }
        meanPos /= vertCount;

        node->metric = localSampler.metricAt(
            meanPos, mesh.verts[xSort.front()].c);
    }
}

void KdTreeSampler::buildGpuBuffers(KdNode* node,
        std::vector<GpuKdNode>& kdNodes) const
{
    GpuKdNode kdNode;

    kdNode.left = -1;
    kdNode.right = -1;
    kdNode.separator = node->separator;
    kdNode.metric = GpuMetric(node->metric);

    size_t kdNodeId = kdNodes.size();
    kdNodes.push_back(kdNode);

    if(node->left != nullptr)
    {
        kdNodes[kdNodeId].left = kdNodes.size();
        buildGpuBuffers(node->left, kdNodes);
    }

    if(node->right != nullptr)
    {
        kdNodes[kdNodeId].right = kdNodes.size();
        buildGpuBuffers(node->right, kdNodes);
    }
}

void KdTreeSampler::meshTree(KdNode* node, Mesh& mesh)
{
    if(node->left == nullptr ||
       node->right == nullptr)
    {
        {
            uint baseVert = mesh.verts.size();
            mesh.verts.push_back(glm::dvec3(node->minBox.x, node->minBox.y, node->minBox.z));
            mesh.verts.push_back(glm::dvec3(node->maxBox.x, node->minBox.y, node->minBox.z));
            mesh.verts.push_back(glm::dvec3(node->maxBox.x, node->maxBox.y, node->minBox.z));
            mesh.verts.push_back(glm::dvec3(node->minBox.x, node->maxBox.y, node->minBox.z));
            mesh.verts.push_back(glm::dvec3(node->minBox.x, node->minBox.y, node->maxBox.z));
            mesh.verts.push_back(glm::dvec3(node->maxBox.x, node->minBox.y, node->maxBox.z));
            mesh.verts.push_back(glm::dvec3(node->maxBox.x, node->maxBox.y, node->maxBox.z));
            mesh.verts.push_back(glm::dvec3(node->minBox.x, node->maxBox.y, node->maxBox.z));

            MeshHex hex(baseVert + 0, baseVert + 1, baseVert + 2, baseVert + 3,
                        baseVert + 4, baseVert + 5, baseVert + 6, baseVert + 7);
            hex.value = glm::sqrt(25.0 / node->metric[0][0]);
            mesh.hexs.push_back(hex);
        }
    }
    else
    {
        meshTree(node->left, mesh);
        meshTree(node->right, mesh);
    }
}
