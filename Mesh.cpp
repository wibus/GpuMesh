#include "Mesh.h"

#include <map>
#include <iostream>

using namespace std;


void Mesh::initialize(
        std::vector<glm::dvec3>& vertices,
        std::vector<Tetrahedron>& tetrahedras)
{
    vert.resize(vertices.size());
    for(int i=0; i<vert.size(); ++i)
    {
        vert[i] = Vertex(vertices[i]);
    }

    tetra.clear();
    for(auto& t : tetrahedras)
    {
        computeCircumSphere(t);
        tetra.push_back(t);
    }
}

void Mesh::compileArrayBuffers(
        std::vector<unsigned int>& indices,
        std::vector<glm::dvec3>& vertices)
{
    int id = -1;
    indices.resize(elemCount());
    for(const auto& t : tetra)
    {
        indices[++id] = t.v[0];
        indices[++id] = t.v[1];
        indices[++id] = t.v[2];

        indices[++id] = t.v[0];
        indices[++id] = t.v[2];
        indices[++id] = t.v[3];

        indices[++id] = t.v[0];
        indices[++id] = t.v[3];
        indices[++id] = t.v[1];

        indices[++id] = t.v[1];
        indices[++id] = t.v[3];
        indices[++id] = t.v[2];
    }

    id = -1;
    vertices.resize(vertCount());
    for(const auto& v : vert)
    {
        vertices[++id] = v.p;
    }

    std::cout << "NV / NT: "
              << tetra.size() << "/" << vert.size() << " = "
              << tetra.size() / (double) vert.size() << endl;
}

void Mesh::insertVertices(const std::vector<glm::dvec3>& vertices)
{
    int idStart = vert.size();
    int idEnd = idStart + vertices.size();
    vert.resize(vert.size() + vertices.size());

    for(int i=idStart; i<idEnd; ++i)
    {
        vert[i] = Vertex(vertices[i]);
    }

    /* Flat implementation
    for(int i=idStart; i<idEnd; ++i)
    {
        insertVertex(i);
        if(((i-idStart) * 1000) % (idEnd-idStart) == 0)
            std::cout << (i-idStart)*100 / (double) (idEnd-idStart) << "% done" << endl;
    }
    //*/

    //* KD Tree implementation
    std::cout << "Building the KD Tree" << endl;
    std::shared_ptr<KdNode> kdRoot = buildKdTree(8);
    std::cout << "Inserting vertices in the mesh" << endl;
    for(int i=idStart; i<idEnd; ++i)
    {
        insertVertexKd(kdRoot, i);

        if(((1+i-idStart) * 10) % (idEnd-idStart) == 0)
            std::cout << (1+i-idStart)*100 / (double) (idEnd-idStart) << "% done" << endl;

        if(((1+i-idStart) * 10) % (idEnd-idStart) == 0)
        {
            printLoads(kdRoot);
        }
    }

    std::cout << "Collecting tetrahedrons" << endl;
    collectTetrahedronsInKd(kdRoot);
    //*/
}

void Mesh::computeCircumSphere(Tetrahedron& tetra)
{
    glm::dvec3 A = vert[tetra.v[0]].p;
    glm::dvec3 B = vert[tetra.v[1]].p;
    glm::dvec3 C = vert[tetra.v[2]].p;
    glm::dvec3 D = vert[tetra.v[3]].p;
    glm::dmat3 S = glm::transpose(
        glm::dmat3(A-B, B-C, C-D));
    glm::dvec3 R(
        (glm::dot(A, A) - glm::dot(B, B)) / 2.0f,
        (glm::dot(B, B) - glm::dot(C, C)) / 2.0f,
        (glm::dot(C, C) - glm::dot(D, D)) / 2.0f);

    double SdetInv = 1.0 / glm::determinant(S);
    double Sx = glm::determinant(glm::dmat3(R, S[1], S[2]));
    double Sy = glm::determinant(glm::dmat3(S[0], R, S[2]));
    double Sz = glm::determinant(glm::dmat3(S[0], S[1], R));

    tetra.circumCenter = glm::dvec3(Sx, Sy, Sz) * SdetInv;
    glm::dvec3 dist = A - tetra.circumCenter;
    tetra.circumRadius2 = glm::dot(dist, dist);
}

void Mesh::insertVertex(int id)
{
    map<Triangle, int> triBuffer;
    const glm::dvec3& v = vert[id].p;

    auto tetraIt = tetra.begin();
    while(tetraIt != tetra.end())
    {
        Tetrahedron& t = *tetraIt;
        glm::dvec3 dist = v - t.circumCenter;
        double len2 = glm::dot(dist, dist);

        if(len2 < t.circumRadius2)
        {
            ++triBuffer[Triangle(t.v[0], t.v[1], t.v[2])];
            ++triBuffer[Triangle(t.v[0], t.v[2], t.v[3])];
            ++triBuffer[Triangle(t.v[0], t.v[3], t.v[1])];
            ++triBuffer[Triangle(t.v[1], t.v[3], t.v[2])];

            tetraIt = tetra.erase(tetraIt);
        }
        else
        {
            ++tetraIt;
        }
    }

    for(const auto& triCount : triBuffer)
    {
        if(triCount.second == 1)
        {
            // TODO: Verify that triangle's orientation are right
            const Triangle& tri = triCount.first;
            Tetrahedron t(id, tri.v[0], tri.v[1], tri.v[2]);
            computeCircumSphere(t);
            tetra.push_back(t);
        }
    }
}

void Mesh::insertVertexKd(const std::shared_ptr<KdNode>& root, int id)
{
    map<Triangle, int> triBuffer;
    const glm::dvec3& v = vert[id].p;

    std::shared_ptr<KdNode> node = root;
    while(node.get() != nullptr)
    {
        auto& nodeTetra = node->tetra;
        auto tetraIt = nodeTetra.begin();
        auto tetraEnd = nodeTetra.end();
        while(tetraIt != tetraEnd)
        {
            const Tetrahedron& t = *tetraIt;
            glm::dvec3 dist = v - t.circumCenter;
            double len2 = glm::dot(dist, dist);

            if(len2 < t.circumRadius2)
            {
                ++triBuffer[Triangle(t.v[0], t.v[1], t.v[2])];
                ++triBuffer[Triangle(t.v[0], t.v[2], t.v[3])];
                ++triBuffer[Triangle(t.v[0], t.v[3], t.v[1])];
                ++triBuffer[Triangle(t.v[1], t.v[3], t.v[2])];
                tetraIt = nodeTetra.erase(tetraIt);
            }
            else
            {
                ++tetraIt;
            }
        }

        if(node->separator != -1)
        {
            const glm::dvec3& axis = node->sepAxis;
            glm::dvec3 vMed = vert[node->separator].p;
            if(glm::dot(v - vMed, axis) < 0)
                node = node->left;
            else
                node = node->right;
        }
        else
        {
            node.reset();
        }
    }

    for(const auto& triCount : triBuffer)
    {
        if(triCount.second == 1)
        {
            // TODO: Verify that triangle's orientation are right
            const Triangle& tri = triCount.first;
            Tetrahedron t(id, tri.v[0], tri.v[1], tri.v[2]);
            computeCircumSphere(t);
            insertTetrahedronKd(root, t);
        }
    }
}

std::shared_ptr<KdNode> Mesh::buildKdTree(int height)
{
    int n = vertCount();
    std::vector<int> vertSorts[3];
    vertSorts[0].resize(n);

    std::iota(vertSorts[0].begin(), vertSorts[0].end(), 0);
    vertSorts[2] = vertSorts[1] = vertSorts[0];

    std::sort(vertSorts[0].begin(), vertSorts[0].end(),
        [this](int a, int b){return vert[a].p.x < vert[b].p.x;});
    for(int i=0; i<n; ++i)
        vert[vertSorts[0][i]].rank[0] = i;

    std::sort(vertSorts[1].begin(), vertSorts[1].end(),
        [this](int a, int b){return vert[a].p.y < vert[b].p.y;});
    for(int i=0; i<n; ++i)
        vert[vertSorts[1][i]].rank[1] = i;

    std::sort(vertSorts[2].begin(), vertSorts[2].end(),
        [this](int a, int b){return vert[a].p.z < vert[b].p.z;});
    for(int i=0; i<n; ++i)
        vert[vertSorts[2][i]].rank[2] = i;

    glm::ivec3 xAxis(1, 0, 0);
    kdRoot.reset(new KdNode());
    buildSubKdTree(
        kdRoot,
        vertSorts,
        xAxis,
        height);

    for(auto t : tetra)
    {
        insertTetrahedronKd(kdRoot, t);
    }

    return kdRoot;
}

void Mesh::buildSubKdTree(
        const std::shared_ptr<KdNode>& node,
        const std::vector<int> sorts[3],
        const glm::ivec3& sepAxis,
        int height)
{
    int count = sorts[0].size();

    if(height != 0 && count > 1)
    {
        int rank = (sepAxis.x + sepAxis.y*2 + sepAxis.z*3) - 1;
        const std::vector<int>& sort = sorts[rank];
        int medianId = sort[count / 2];
        const Vertex& vMed = vert[medianId];

        int leftSize = (count+1) / 2;
        std::vector<int> leftSorts[3];
        leftSorts[0].reserve(leftSize);
        leftSorts[1].reserve(leftSize);
        leftSorts[2].reserve(leftSize);

        int rightSize = count - leftSize;
        std::vector<int> rightSorts[3];
        rightSorts[0].reserve(rightSize);
        rightSorts[1].reserve(rightSize);
        rightSorts[2].reserve(rightSize);

        for(int r=0; r<3; ++r)
        {
            for(int i=0; i<count; ++i)
            {
                int vId = sorts[r][i];
                if(vert[vId].rank[rank] < vMed.rank[rank])
                    leftSorts[r].push_back(vId);
                else
                    rightSorts[r].push_back(vId);
            }
        }

        glm::ivec3 nextSepAxis(sepAxis.z, sepAxis.x, sepAxis.y);

        node->left.reset(new KdNode());
        buildSubKdTree(
            node->left,
            leftSorts,
            nextSepAxis,
            height-1);

        node->right.reset(new KdNode());
        buildSubKdTree(
            node->right,
            rightSorts,
            nextSepAxis,
            height-1);

        node->sepAxis = glm::dvec3(sepAxis);
        node->separator = medianId;
    }
    else
    {
        node->separator = -1;
    }
}

void Mesh::insertTetrahedronKd(const std::shared_ptr<KdNode>& root,
                               const Tetrahedron& tet)
{
    double radius2 = tet.circumRadius2;
    glm::dvec3 center = tet.circumCenter;
    std::shared_ptr<KdNode> node = root;

    while(true)
    {
        if(node->separator != -1)
        {
            const glm::dvec3& axis = node->sepAxis;
            glm::dvec3 vMed = vert[node->separator].p;
            double dist = glm::dot(center - vMed, axis);
            if(dist*dist < radius2)
            {
                node->tetra.push_back(tet);
                return;
            }
            else
            {
                if(dist < 0)
                    node = node->left;
                else
                    node = node->right;
            }
        }
        else
        {
            node->tetra.push_back(tet);
            return;
        }
    }
}

void Mesh::collectTetrahedronsInKd(const std::shared_ptr<KdNode>& root)
{
    std::vector<std::shared_ptr<KdNode>> nodes;
    nodes.push_back(root);
    tetra.clear();

    for(int i=0; i<nodes.size(); ++i)
    {
        std::shared_ptr<KdNode> n = nodes[i];

        if(n->separator != -1)
        {
            nodes.push_back(n->left);
            nodes.push_back(n->right);
        }

        for(const auto& t : n->tetra)
        {
            tetra.push_back(t);
        }
    }
}

void Mesh::printLoads(const std::shared_ptr<KdNode> root)
{
    std::vector<std::shared_ptr<KdNode>> nodes;
    nodes.push_back(root);

    int i=0;
    int levelId = 0;
    int levelSize = 1;
    int nextLevel = levelSize;

    int sum = 0;
    int total = 0;
    double rMoy = 0;

    while(i < nodes.size())
    {
        if(nodes[i]->separator != -1)
        {
            nodes.push_back(nodes[i]->left);
            nodes.push_back(nodes[i]->right);
        }

        sum += nodes[i]->tetra.size();

        for(auto t : nodes[i]->tetra)
        {
            rMoy += glm::sqrt(t.circumRadius2);
        }

        ++i;
        if(i == nextLevel)
        {
            rMoy /= sum;
            total += sum;
            cout << levelId << ": " <<
                    "tetCount= " << sum << "\t" <<
                    "meanRad= " << rMoy << "\t" <<
                    "meanLoad=" << sum / (double) levelSize <<
                    endl;

            levelId += 1;
            levelSize *= 2;
            nextLevel += levelSize;

            sum = 0;
            rMoy = 0;
        }
    }

    cout << ":: Total = " << total << endl << endl;
}
