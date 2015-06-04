#ifndef GPUMESH_MESH
#define GPUMESH_MESH

#include <list>
#include <vector>
#include <memory>
#include <algorithm>

#include <GLM/glm.hpp>


struct Vertex
{
    Vertex() : p(), rank{-1, -1, -1} {}
    Vertex(const glm::dvec3 pos) :
        p(pos)
    {}

    glm::dvec3 p;
    int rank[3];
};

struct Triangle
{
    Triangle(int v0, int v1, int v2)
    {
        if(v0 < v1)
        {
            if(v0 < v2)
            {
                if(v1 < v2)
                {
                    v[0] = v0;
                    v[1] = v1;
                    v[2] = v2;
                }
                else // v2 < v1
                {
                    v[0] = v0;
                    v[1] = v2;
                    v[2] = v1;
                }
            }
            else // v2 < v0
            {
                v[0] = v2;
                v[1] = v0;
                v[2] = v1;
            }
        }
        else // v1 < v0
        {
            if(v1 < v2)
            {
                if(v0 < v2)
                {
                    v[0] = v1;
                    v[1] = v0;
                    v[2] = v2;
                }
                else // v2 < v0
                {
                    v[0] = v1;
                    v[1] = v2;
                    v[2] = v0;
                }
            }
            else // v2 < v1
            {
                v[0] = v2;
                v[1] = v1;
                v[2] = v0;
            }
        }
    }

    bool operator< (const Triangle& t) const
    {
        if(v[0] < t.v[0])
            return true;
        else if(v[0] > t.v[0])
            return false;

        if(v[1] < t.v[1])
            return true;
        else if(v[1] > t.v[1])
            return false;

        if(v[2] < t.v[2])
            return true;

        return false;
    }

    int v[3];
};

struct Tetrahedron
{
    Tetrahedron(int v0, int v1, int v2, int v3) :
        v{v0, v1, v2, v3}
    {
    }

    int v[4];
    double circumRadius2;
    glm::dvec3 circumCenter;
};


struct KdNode
{
    std::shared_ptr<KdNode> left;
    std::shared_ptr<KdNode> right;
    std::list<Tetrahedron> tetra;
    glm::dvec3 sepAxis;
    int separator;
};

class Mesh
{
public:
    inline unsigned int vertCount() const
    {
        return vert.size();
    }

    inline unsigned int elemCount() const
    {
        return tetra.size() * 12;
    }

    void initialize(
            std::vector<glm::dvec3>& vertices,
            std::vector<Tetrahedron>& tetrahedras);

    void compileArrayBuffers(
            std::vector<unsigned int>& indices,
            std::vector<glm::dvec3>& vertices);

    void insertVertices(const std::vector<glm::dvec3>& vertices);

    std::vector<Vertex> vert;
    std::list<Tetrahedron> tetra;
    std::shared_ptr<KdNode> kdRoot;

private:
    void computeCircumSphere(Tetrahedron& tetra);

    void insertVertex(int id);

    std::shared_ptr<KdNode> buildKdTree(int height);
    void buildSubKdTree(const std::shared_ptr<KdNode>& node,
                        const std::vector<int> sorts[3],
                        const glm::ivec3& sepAxis,
                        int height);

    void insertVertexKd(const std::shared_ptr<KdNode>& root, int id);
    void insertTetrahedronKd(const std::shared_ptr<KdNode>& root,
                             const Tetrahedron& tet);

    void collectTetrahedronsInKd(const std::shared_ptr<KdNode>& root);
    void printLoads(const std::shared_ptr<KdNode> root);
};


#endif // GPUMESH_MESH
