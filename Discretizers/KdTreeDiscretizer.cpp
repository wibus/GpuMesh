#include "KdTreeDiscretizer.h"

#include <algorithm>

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/Mesh.h"

using namespace cellar;


struct KdNode
{
    KdNode() :
        left(nullptr),
        right(nullptr),
        value(0)
    {}

    ~KdNode()
    {
        delete left;
        delete right;
    }

    KdNode* left;
    KdNode* right;

    glm::dvec3 separator;
    glm::dvec3 minBox;
    glm::dvec3 maxBox;

    double value;
};

KdTreeDiscretizer::KdTreeDiscretizer() :
    _gridMesh(new Mesh())
{
    _gridMesh->modelName = "Kd-Tree Discretization Grid";
}

KdTreeDiscretizer::~KdTreeDiscretizer()
{

}

std::shared_ptr<Mesh> KdTreeDiscretizer::gridMesh() const
{
    return _gridMesh;
}

void KdTreeDiscretizer::discretize(const Mesh& mesh, const glm::ivec3& gridSize)
{
    size_t vertCount = mesh.verts.size();
    int height = glm::min((int)std::log2(vertCount), gridSize.x);

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

    glm::dvec3 minBounds, maxBounds;
    boundingBox(mesh, minBounds, maxBounds);

    KdNode root;
    build(&root, height, mesh, minBounds, maxBounds, xSort, ySort, zSort);

    _gridMesh->clear();
    meshTree(&root, *_gridMesh);
    _gridMesh->compileTopology();
}

void KdTreeDiscretizer::build(
        KdNode* node,
        int height,
        const Mesh& mesh,
        const glm::dvec3& minBox,
        const glm::dvec3& maxBox,
        std::vector<uint>& xSort,
        std::vector<uint>& ySort,
        std::vector<uint>& zSort)
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

        if(vertCount % 2 == 0)
            node->separator = (mesh.verts[sepL].p + mesh.verts[sepR].p) / 2.0;
        else
            node->separator = mesh.verts[sepR].p;


        // Distribute vertices around the separator
        std::vector<uint> xSortL;
        std::vector<uint> xSortR;
        std::vector<uint> ySortL;
        std::vector<uint> ySortR;
        std::vector<uint> zSortL;
        std::vector<uint> zSortR;
        for(size_t v=0; v < vertCount; ++v)
        {
            double xDist = glm::dot(mesh.verts[xSort[v]].p - node->separator, axis);
            if(xDist <= 0.0)
                xSortL.push_back(xSort[v]);
            if(xDist >= 0.0)
                xSortR.push_back(xSort[v]);

            double yDist = glm::dot(mesh.verts[ySort[v]].p - node->separator, axis);
            if(yDist <= 0.0)
                ySortL.push_back(ySort[v]);
            if(yDist >= 0.0)
                ySortR.push_back(ySort[v]);

            double zDist = glm::dot(mesh.verts[zSort[v]].p - node->separator, axis);
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
        glm::dvec3 maxBoxL = glm::mix(maxBox, node->separator, axis);

        glm::dvec3 minBoxR = glm::mix(minBox, node->separator, axis);
        glm::dvec3 maxBoxR = maxBox;

        // Build children nodes
        node->left = new KdNode();
        build(node->left, height-1, mesh, minBoxL, maxBoxL, xSortL, ySortL, zSortL);

        node->right = new KdNode();
        build(node->right, height-1, mesh, minBoxR, maxBoxR, xSortR, ySortR, zSortR);

        // Compute node's value from children's
        node->value = (node->left->value + node->right->value) / 2.0;
    }
    else
    {
        assert(vertCount > 0);

        // Compute node's value from vertices'
        if(vertCount > 0)
        {
            for(size_t v=0; v < vertCount; ++v)
                node->value += vertValue(mesh, xSort[v]);
            node->value /= vertCount;
        }
        else
        {
            node->value = -1.0;
        }
    }
}

void KdTreeDiscretizer::meshTree(KdNode* node, Mesh& mesh)
{
    if(node->left == nullptr ||
       node->right == nullptr)
    {
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
        hex.value = node->value;
        mesh.hexs.push_back(hex);
    }
    else
    {
        meshTree(node->left, mesh);
        meshTree(node->right, mesh);
    }
}
