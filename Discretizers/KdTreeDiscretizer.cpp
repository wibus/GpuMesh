#include "KdTreeDiscretizer.h"

#include <algorithm>
#include <numeric>

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/GpuMesh.h"

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

struct GpuKdNode
{
    GpuKdNode() :
        left(-1),
        right(-1),
        tetBeg(0),
        tetEnd(0)
    {}

    GLint left;
    GLint right;

    GLuint tetBeg;
    GLuint tetEnd;

    glm::vec4 separator;
};


KdTreeDiscretizer::KdTreeDiscretizer() :
    AbstractDiscretizer("Kd-Tree", ":/glsl/compute/Discretizing/KdTree.glsl"),
    _debugMesh(new Mesh()),
    _kdNodesSsbo(0),
    _kdTetsSsbo(0),
    _kdMetricsSsbo(0),
    _metricAtSub(0)
{
}

KdTreeDiscretizer::~KdTreeDiscretizer()
{
}

bool KdTreeDiscretizer::isMetricWise() const
{
    return true;
}

void KdTreeDiscretizer::installPlugin(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    AbstractDiscretizer::installPlugin(mesh, program);
}

void KdTreeDiscretizer::setPluginUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    AbstractDiscretizer::setPluginUniforms(mesh, program);
}

void KdTreeDiscretizer::setupPluginExecution(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{
    AbstractDiscretizer::setupPluginExecution(mesh, program);

    GLuint kdNodes   = mesh.bufferBinding(EBufferBinding::KD_NODES_BUFFER_BINDING);
    GLuint kdTets    = mesh.bufferBinding(EBufferBinding::KD_TETS_BUFFER_BINDING);
    GLuint kdMetrics = mesh.bufferBinding(EBufferBinding::KD_METRICS_BUFFER_BINDING);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kdNodes,   _kdNodesSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kdTets,    _kdTetsSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kdMetrics, _kdMetricsSsbo);

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_metricAtSub);
}

void KdTreeDiscretizer::discretize(const Mesh& mesh, int density)
{
    // Clear resources
    _debugMesh->clear();
    _debugMesh->verts = mesh.verts;

    _vertMetrics.clear();
    _vertMetrics.shrink_to_fit();

    glDeleteBuffers(1, &_kdNodesSsbo);
    _kdNodesSsbo = 0;
    glDeleteBuffers(1, &_kdTetsSsbo);
    _kdTetsSsbo = 0;
    glDeleteBuffers(1, &_kdMetricsSsbo);
    _kdMetricsSsbo = 0;

    _rootNode.reset();


    // Break prisms and hex into tetrahedra
    std::vector<MeshTet> tets;
    tetrahedrizeMesh(mesh, tets);

    // Compute Kd Tree depth
    size_t vertCount = mesh.verts.size();
    int height = (int)std::log2(vertCount/density);

    getLog().postMessage(new Message('I', false,
        "Discretizing mesh metric in a Kd-Tree",
         "KdTreeDiscretizer"));
    getLog().postMessage(new Message('I', false,
        "Maximum Kd-Tree's depth: " + std::to_string(height),
        "KdTreeDiscretizer"));

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


    // Fill Discretizer's data strucutres
    glm::dvec3 minBounds, maxBounds;
    boundingBox(mesh, minBounds, maxBounds);

    _rootNode.reset(new KdNode());
    build(_rootNode.get(), height, mesh,
          minBounds, maxBounds,
          xSort, ySort, zSort,
          tets);


    _vertMetrics.reserve(vertCount);
    for(size_t v=0; v < vertCount; ++v)
        _vertMetrics.push_back(vertMetric(mesh.verts[v]));


    // Update SSBOs
    std::vector<GpuTet> gpuKdTets;
    std::vector<GpuKdNode> gpuKdNodes;
    buildGpuBuffers(_rootNode.get(), gpuKdNodes, gpuKdTets);

    glGenBuffers(1, &_kdNodesSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _kdNodesSsbo);
    size_t kdNodesSize = sizeof(decltype(gpuKdNodes.front())) * gpuKdNodes.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, kdNodesSize, gpuKdNodes.data(), GL_STATIC_DRAW);
    gpuKdNodes.clear();
    gpuKdNodes.shrink_to_fit();

    glGenBuffers(1, &_kdTetsSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _kdTetsSsbo);
    size_t kdTetsSize = sizeof(decltype(gpuKdTets.front())) * gpuKdTets.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, kdTetsSize, gpuKdTets.data(), GL_STATIC_DRAW);
    gpuKdTets.clear();
    gpuKdTets.shrink_to_fit();


    std::vector<glm::mat4> gpuKdMetrics;
    gpuKdMetrics.reserve(_vertMetrics.size());
    for(const auto& metric : _vertMetrics)
        gpuKdMetrics.push_back(glm::mat4(metric));

    glGenBuffers(1, &_kdMetricsSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _kdMetricsSsbo);
    size_t kdMetricsSize = sizeof(decltype(gpuKdMetrics.front())) * gpuKdMetrics.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, kdMetricsSize, gpuKdMetrics.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

inline bool KdTreeDiscretizer::tetParams(
        const std::vector<MeshVert>& verts,
        const MeshTet& tet,
        const glm::dvec3& p,
        double coor[4])
{
    // ref : https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Barycentric_coordinates_on_tetrahedra

    const glm::dvec3& vp0 = verts[tet.v[0]].p;
    const glm::dvec3& vp1 = verts[tet.v[1]].p;
    const glm::dvec3& vp2 = verts[tet.v[2]].p;
    const glm::dvec3& vp3 = verts[tet.v[3]].p;

    glm::dmat3 T(vp0 - vp3, vp1 - vp3, vp2 - vp3);

    glm::dvec3 y = glm::inverse(T) * (p - vp3);
    coor[0] = y[0];
    coor[1] = y[1];
    coor[2] = y[2];
    coor[3] = 1.0 - (y[0] + y[1] + y[2]);

    const double EPSILON_IN = -1e-8;
    bool isIn = (coor[0] >= EPSILON_IN && coor[1] >= EPSILON_IN &&
                 coor[2] >= EPSILON_IN && coor[3] >= EPSILON_IN);
    return isIn;
}

Metric KdTreeDiscretizer::metricAt(
        const glm::dvec3& position) const
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
        double coor[4];
        size_t tetCount = node->tets.size();
        const auto& verts = _debugMesh->verts;
        for(size_t t=0; t < tetCount; ++t)
        {
            const MeshTet& tet = node->tets[t];
            if(tetParams(verts, tet, position, coor))
            {
                Metric m = coor[0] * _vertMetrics[tet.v[0]] +
                           coor[1] * _vertMetrics[tet.v[1]] +
                           coor[2] * _vertMetrics[tet.v[2]] +
                           coor[3] * _vertMetrics[tet.v[3]];
                return m;
            }
        }

        // Outside of node's tets
        uint nearestVert = 0;
        double nearestDist = INFINITY;
        for(size_t t=0; t < tetCount; ++t)
        {
            double distance;
            const MeshTet& tet = node->tets[t];
            if((distance = glm::distance(position, verts[tet.v[0]].p)) < nearestDist)
                { nearestVert = tet.v[0]; nearestDist = distance; }
            if((distance = glm::distance(position, verts[tet.v[1]].p)) < nearestDist)
                { nearestVert = tet.v[1]; nearestDist = distance; }
            if((distance = glm::distance(position, verts[tet.v[2]].p)) < nearestDist)
                { nearestVert = tet.v[2]; nearestDist = distance; }
            if((distance = glm::distance(position, verts[tet.v[3]].p)) < nearestDist)
                { nearestVert = tet.v[3]; nearestDist = distance; }
        }

        return _vertMetrics[nearestVert];
    }
    else return METRIC_ERROR;
}

void KdTreeDiscretizer::releaseDebugMesh()
{
    auto verts = _debugMesh->verts;
    _debugMesh->clear();
    _debugMesh->verts = verts;
}

const Mesh& KdTreeDiscretizer::debugMesh()
{
    if(_debugMesh->tets.empty() &&
       !_debugMesh->verts.empty())
    {
        if(_rootNode.get() != nullptr)
        {
            meshTree(_rootNode.get(), *_debugMesh);

            _debugMesh->modelName = "Kd-Tree Discretization Mesh";
            _debugMesh->compileTopology();
        }
    }

    return *_debugMesh;
}

void KdTreeDiscretizer::build(
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

void KdTreeDiscretizer::buildGpuBuffers(
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

void KdTreeDiscretizer::meshTree(KdNode* node, Mesh& mesh)
{
    static int cellId = 0;

    if(node->left == nullptr ||
       node->right == nullptr)
    {
        {
            cellId = 0;
            mesh.tets.insert(mesh.tets.end(), node->tets.begin(), node->tets.end());

            uint baseVert = mesh.verts.size();
            mesh.verts.push_back(glm::dvec3(node->minBox.x, node->minBox.y, node->minBox.z));
            mesh.verts.push_back(glm::dvec3(node->maxBox.x, node->minBox.y, node->minBox.z));
            mesh.verts.push_back(glm::dvec3(node->minBox.x, node->maxBox.y, node->minBox.z));
            mesh.verts.push_back(glm::dvec3(node->maxBox.x, node->maxBox.y, node->minBox.z));
            mesh.verts.push_back(glm::dvec3(node->minBox.x, node->minBox.y, node->maxBox.z));
            mesh.verts.push_back(glm::dvec3(node->maxBox.x, node->minBox.y, node->maxBox.z));
            mesh.verts.push_back(glm::dvec3(node->minBox.x, node->maxBox.y, node->maxBox.z));
            mesh.verts.push_back(glm::dvec3(node->maxBox.x, node->maxBox.y, node->maxBox.z));

            MeshHex hex(baseVert + 0, baseVert + 1, baseVert + 2, baseVert + 3,
                        baseVert + 4, baseVert + 5, baseVert + 6, baseVert + 7);
            hex.value = 1.0;
            mesh.hexs.push_back(hex);
        }
    }
    else
    {
        meshTree(node->left, mesh);
        meshTree(node->right, mesh);
    }
}
