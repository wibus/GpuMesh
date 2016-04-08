#include "KdTreeSampler.h"

#include <algorithm>
#include <numeric>

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/GpuMesh.h"
#include "DataStructures/Tetrahedralizer.h"

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

    std::vector<MeshTet> tets;

    glm::dvec4 separator;
    glm::dvec3 minBox;
    glm::dvec3 maxBox;
};


// CUDA Drivers Interface
void installCudaKdTreeSampler();
void updateCudaKdTets(
        const std::vector<GpuTet>& kdTetsBuff);
void updateCudaKdNodes(
        const std::vector<GpuKdNode>& kdNodesBuff);
void updateCudaRefVerts(
        const std::vector<GpuVert>& refVertsBuff);
void updateCudaRefMetrics(
        const std::vector<glm::mat4>& refMetricsBuff);


KdTreeSampler::KdTreeSampler() :
    AbstractSampler("Kd-Tree", ":/glsl/compute/Sampling/KdTree.glsl", installCudaKdTreeSampler),
    _debugMesh(new Mesh()),
    _kdTetsSsbo(0),
    _kdNodesSsbo(0),
    _refVertsSsbo(0),
    _refMetricsSsbo(0),
    _metricAtSub(0)
{
}

KdTreeSampler::~KdTreeSampler()
{
    glDeleteBuffers(1, &_kdTetsSsbo);
    _kdTetsSsbo = 0;
    glDeleteBuffers(1, &_kdNodesSsbo);
    _kdNodesSsbo = 0;
    glDeleteBuffers(1, &_refVertsSsbo);
    _refVertsSsbo = 0;
    glDeleteBuffers(1, &_refMetricsSsbo);
    _refMetricsSsbo = 0;
}

bool KdTreeSampler::isMetricWise() const
{
    return true;
}

void KdTreeSampler::initialize()
{
    if(_kdTetsSsbo == 0)
        glGenBuffers(1, &_kdTetsSsbo);

    if(_kdNodesSsbo == 0)
        glGenBuffers(1, &_kdNodesSsbo);

    if(_refVertsSsbo == 0)
        glGenBuffers(1, &_refVertsSsbo);

    if(_refMetricsSsbo == 0)
        glGenBuffers(1, &_refMetricsSsbo);
}

void KdTreeSampler::installPlugin(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    AbstractSampler::installPlugin(mesh, program);

    GLuint kdTets     = mesh.bufferBinding(EBufferBinding::KD_TETS_BUFFER_BINDING);
    GLuint kdNodes    = mesh.bufferBinding(EBufferBinding::KD_NODES_BUFFER_BINDING);
    GLuint refVerts   = mesh.bufferBinding(EBufferBinding::REF_VERTS_BUFFER_BINDING);
    GLuint refMetrics = mesh.bufferBinding(EBufferBinding::REF_METRICS_BUFFER_BINDING);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kdTets,     _kdTetsSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kdNodes,    _kdNodesSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, refVerts,   _refVertsSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, refMetrics, _refMetricsSsbo);
}

void KdTreeSampler::setupPluginExecution(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{
    AbstractSampler::setupPluginExecution(mesh, program);

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_metricAtSub);
}

void KdTreeSampler::setReferenceMesh(const Mesh& mesh)
{
    size_t vertCount = mesh.verts.size();

    // Clear resources
    _debugMesh.reset();

    _rootNode.reset();
    _refVerts = mesh.verts;
    _refVerts.shrink_to_fit();
    _refMetrics.resize(mesh.verts.size());
    for(size_t vId=0; vId < vertCount; ++vId)
        _refMetrics[vId] = vertMetric(mesh, vId);
    _refMetrics.shrink_to_fit();


    // Break prisms and hex into tetrahedra
    std::vector<MeshTet> tets;
    tetrahedrize(tets, mesh);

    // Compute Kd Tree depth
    int height = (int)std::log2(vertCount/2);

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


    // Fill Sampler's data strucutres
    glm::dvec3 minBounds, maxBounds;
    boundingBox(mesh, minBounds, maxBounds);

    _rootNode.reset(new KdNode());
    build(_rootNode.get(), height, mesh,
          minBounds, maxBounds,
          xSort, ySort, zSort,
          tets);


    // Build GPU Buffers
    std::vector<GpuTet> gpuKdTets;
    std::vector<GpuKdNode> gpuKdNodes;
    buildGpuBuffers(_rootNode.get(), gpuKdNodes, gpuKdTets);

    updateCudaKdTets(gpuKdTets);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _kdTetsSsbo);
    size_t kdTetsSize = sizeof(decltype(gpuKdTets.front())) * gpuKdTets.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, kdTetsSize, gpuKdTets.data(), GL_STREAM_COPY);
    gpuKdTets.clear();
    gpuKdTets.shrink_to_fit();

    updateCudaKdNodes(gpuKdNodes);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _kdNodesSsbo);
    size_t kdNodesSize = sizeof(decltype(gpuKdNodes.front())) * gpuKdNodes.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, kdNodesSize, gpuKdNodes.data(), GL_STREAM_COPY);
    gpuKdNodes.clear();
    gpuKdNodes.shrink_to_fit();


    // Reference Mesh Vertices
    std::vector<GpuVert> gpuRefVerts;
    gpuRefVerts.reserve(_refVerts.size());
    for(const auto& v : _refVerts)
        gpuRefVerts.push_back(GpuVert(v));

    updateCudaRefVerts(gpuRefVerts);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _refVertsSsbo);
    size_t refVertsSize = sizeof(decltype(gpuRefVerts.front())) * gpuRefVerts.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, refVertsSize, gpuRefVerts.data(), GL_STREAM_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    // Reference Mesh Metrics
    std::vector<glm::mat4> gpuRefMetrics;
    gpuRefMetrics.reserve(_refMetrics.size());
    for(const auto& metric : _refMetrics)
        gpuRefMetrics.push_back(glm::mat4(metric));

    updateCudaRefMetrics(gpuRefMetrics);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _refMetricsSsbo);
    size_t refMetricsSize = sizeof(decltype(gpuRefMetrics.front())) * gpuRefMetrics.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, refMetricsSize, gpuRefMetrics.data(), GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

Metric KdTreeSampler::metricAt(
        const glm::dvec3& position,
        uint cacheId) const
{
    const Metric METRIC_ERROR(0.0);

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
        size_t nodeSmallestIdx = 0;
        double nodeSmallestVal = -1/0.0;
        double nodeSmallestCoor[4];

        double coor[4];
        size_t tetCount = node->tets.size();
        for(size_t t=0; t < tetCount; ++t)
        {
            const MeshTet& tet = node->tets[t];
            if(tetParams(_refVerts, tet, position, coor))
            {
                return coor[0] * _refMetrics[tet.v[0]] +
                       coor[1] * _refMetrics[tet.v[1]] +
                       coor[2] * _refMetrics[tet.v[2]] +
                       coor[3] * _refMetrics[tet.v[3]];
            }
            else
            {
                double tetSmallest = 0.0;
                if(coor[0] < tetSmallest) tetSmallest = coor[0];
                if(coor[1] < tetSmallest) tetSmallest = coor[1];
                if(coor[2] < tetSmallest) tetSmallest = coor[2];
                if(coor[3] < tetSmallest) tetSmallest = coor[3];

                if(tetSmallest > nodeSmallestVal)
                {
                    nodeSmallestIdx = t;
                    nodeSmallestVal = tetSmallest;
                    nodeSmallestCoor[0] = coor[0];
                    nodeSmallestCoor[1] = coor[1];
                    nodeSmallestCoor[2] = coor[2];
                    nodeSmallestCoor[3] = coor[3];
                }
            }
        }


        // Clamp coordinates for project
        double coorSum = 0.0;
        if(nodeSmallestCoor[0] < 0.0)
            nodeSmallestCoor[0] = 0.0;
        coorSum += nodeSmallestCoor[0];
        if(nodeSmallestCoor[1] < 0.0)
            nodeSmallestCoor[1] = 0.0;
        coorSum += nodeSmallestCoor[1];
        if(nodeSmallestCoor[2] < 0.0)
            nodeSmallestCoor[2] = 0.0;
        coorSum += nodeSmallestCoor[2];
        if(nodeSmallestCoor[3] < 0.0)
            nodeSmallestCoor[3] = 0.0;
        coorSum += nodeSmallestCoor[3];

        nodeSmallestCoor[0] /= coorSum;
        nodeSmallestCoor[1] /= coorSum;
        nodeSmallestCoor[2] /= coorSum;
        nodeSmallestCoor[3] /= coorSum;

        // Return projected metric
        const MeshTet& tet = node->tets[nodeSmallestIdx];
        return nodeSmallestCoor[0] * _refMetrics[tet.v[0]] +
               nodeSmallestCoor[1] * _refMetrics[tet.v[1]] +
               nodeSmallestCoor[2] * _refMetrics[tet.v[2]] +
               nodeSmallestCoor[3] * _refMetrics[tet.v[3]];
    }
    else return METRIC_ERROR;
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
        const glm::dvec3& minBox,
        const glm::dvec3& maxBox,
        std::vector<uint>& xSort,
        std::vector<uint>& ySort,
        std::vector<uint>& zSort,
        std::vector<MeshTet>& tets)
{
    node->minBox = minBox;
    node->maxBox = maxBox;

    size_t vertCount = xSort.size();
    if(height > 0 && vertCount >= 2)
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

        std::vector<MeshTet> tetsL;
        std::vector<MeshTet> tetsR;
        size_t tetCount = tets.size();
        for(size_t t=0; t < tetCount; ++t)
        {
            const MeshTet& tet = tets[t];

            bool inL = false;
            bool inR = false;

            for(int i=0; i < 4; ++i)
            {
                double vDist = glm::dot(mesh.verts[tet.v[i]].p, axis) - sepVal;

                if(vDist < 0.0)
                {
                    if(!inL)
                    {
                        inL = true;
                        tetsL.push_back(tet);
                        if(inR) break;
                    }
                }
                else if(vDist > 0.0)
                {
                    if(!inR)
                    {
                        inR = true;
                        tetsR.push_back(tet);
                        if(inL) break;
                    }
                }
                else
                {
                    tetsL.push_back(tet);
                    tetsR.push_back(tet);
                    break;
                }
            }
        }

        // Deallocate sorted vertices vectors
        xSort.clear();
        xSort.shrink_to_fit();
        ySort.clear();
        ySort.shrink_to_fit();
        zSort.clear();
        zSort.shrink_to_fit();
        tets.clear();
        tets.shrink_to_fit();


        // Child bounding boxes
        glm::dvec3 minBoxL = minBox;
        glm::dvec3 maxBoxL = glm::mix(maxBox, sepPos, axis);

        glm::dvec3 minBoxR = glm::mix(minBox, sepPos, axis);
        glm::dvec3 maxBoxR = maxBox;

        // Build children nodes
        node->left = new KdNode();
        build(node->left,  height-1, mesh, minBoxL, maxBoxL, xSortL, ySortL, zSortL, tetsL);

        node->right = new KdNode();
        build(node->right, height-1, mesh, minBoxR, maxBoxR, xSortR, ySortR, zSortR, tetsR);
    }
    else
    {
        assert(vertCount > 0);
        node->tets = tets;

        node->separator.w = 0.0;
    }
}

void KdTreeSampler::buildGpuBuffers(
        KdNode* node,
        std::vector<GpuKdNode>& kdNodes,
        std::vector<GpuTet>& kdTets)
{
    GpuKdNode kdNode;

    kdNode.left = -1;
    kdNode.right = -1;
    kdNode.separator = node->separator;

    kdNode.tetBeg = kdTets.size();
    size_t tetCount = node->tets.size();
    for(size_t t=0; t < tetCount; ++t)
        kdTets.push_back(node->tets[t]);
    kdNode.tetEnd = kdTets.size();


    size_t kdNodeId = kdNodes.size();
    kdNodes.push_back(kdNode);

    if(node->left != nullptr)
    {
        kdNodes[kdNodeId].left = kdNodes.size();
        buildGpuBuffers(node->left, kdNodes, kdTets);
    }

    if(node->right != nullptr)
    {
        kdNodes[kdNodeId].right = kdNodes.size();
        buildGpuBuffers(node->right, kdNodes, kdTets);
    }
}

void KdTreeSampler::meshTree(KdNode* node, Mesh& mesh)
{
    static int cellId = 0;

    if(node->left == nullptr ||
       node->right == nullptr)
    {
        {
            cellId = 0;

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
            hex.value = metricAt((node->minBox + node->maxBox) / 2.0, -1)[0][0];
            mesh.hexs.push_back(hex);
        }
    }
    else
    {
        meshTree(node->left, mesh);
        meshTree(node->right, mesh);
    }
}
