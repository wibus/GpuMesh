#include "KdTreeDiscretizer.h"

#include <algorithm>

#include "DataStructures/Mesh.h"


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

    glm::dvec3 minBox;
    glm::dvec3 maxBox;
    glm::dvec3 separator;

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

    KdNode* root = new KdNode();
    build(root, gridSize.x, mesh, minBounds, maxBounds, xSort, ySort, zSort);

    _gridMesh->clear();
    meshTree(root, *_gridMesh);
    _gridMesh->compileTopology();
}

void KdTreeDiscretizer::build(
        KdNode* node,
        int height,
        const Mesh& mesh,
        const glm::dvec3& minBox,
        const glm::dvec3& maxBox,
        const std::vector<uint>& xSort,
        const std::vector<uint>& ySort,
        const std::vector<uint>& zSort)
{
    node->minBox = minBox;
    node->maxBox = maxBox;

    size_t vertCount = xSort.size();
    if(vertCount >= 2 && height > 0)
    {
        size_t childVertCount = (vertCount + 1) / 2;

        glm::dvec3 extent = maxBox - minBox;

        uint sep = xSort[childVertCount];
        glm::dvec3 axis = glm::dvec3(1, 0, 0);
        if(extent.y > extent.x && extent.y > extent.z)
        {
            sep = ySort[childVertCount];
            axis = glm::dvec3(0, 1, 0);
        }
        else if(extent.z > extent.x && extent.z > extent.y)
        {
            sep = zSort[childVertCount];
            axis = glm::dvec3(0, 0, 1);
        }
        node->separator = mesh.verts[sep];

        glm::dvec3 minBoxL = minBox;
        glm::dvec3 maxBoxL = glm::mix(maxBox, node->separator, axis);

        glm::dvec3 minBoxR = glm::mix(minBox, node->separator, axis);
        glm::dvec3 maxBoxR = maxBox;

        std::vector<uint> xSortL;
        std::vector<uint> xSortR;
        std::vector<uint> ySortL;
        std::vector<uint> ySortR;
        std::vector<uint> zSortL;
        std::vector<uint> zSortR;
        for(size_t v=0; v < vertCount; ++v)
        {
            if(glm::dot(mesh.verts[xSort[v]].p - node->separator, axis) < 0.0)
                xSortL.push_back(xSort[v]);
            else
                xSortR.push_back(xSort[v]);

            if(glm::dot(mesh.verts[ySort[v]].p - node->separator, axis) < 0.0)
                ySortL.push_back(ySort[v]);
            else
                ySortR.push_back(ySort[v]);

            if(glm::dot(mesh.verts[zSort[v]].p - node->separator, axis) < 0.0)
                zSortL.push_back(zSort[v]);
            else
                zSortR.push_back(zSort[v]);
        }

        node->left = new KdNode();
        build(node->left, height-1, mesh, minBoxL, maxBoxL, xSortL, ySortL, zSortL);

        node->right = new KdNode();
        build(node->right, height-1, mesh, minBoxR, maxBoxR, xSortR, ySortR, zSortR);

        node->value = (node->left->value + node->right->value) / 2.0;
    }
    else
    {
        for(size_t v=0; v < vertCount; ++v)
            node->value += vertValue(mesh, xSort[v]);
        node->value /= vertCount;
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
