#ifndef GPUMESH_MESH
#define GPUMESH_MESH

#include <unordered_set>
#include <list>
#include <vector>
#include <memory>
#include <algorithm>

#include <GLM/glm.hpp>


struct Tetrahedron;

struct Vertex
{
    Vertex() {}
    Vertex(const glm::dvec3 pos) :
        p(pos), flag(false)
    {}

    glm::dvec3 p;
    std::unordered_set<Tetrahedron*> tetra;
    bool isBoundary;
    bool flag;
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

    bool operator==(const Triangle& t) const
    {
        return v[0] == t.v[0] &&
               v[1] == t.v[1] &&
               v[2] == t.v[2];
    }

    int v[3];
};

namespace std
{
    template<>
    struct hash<Triangle>
    {
        size_t operator() (const Triangle& t) const
        {
            return t.v[0] * (t.v[1] * 1000001 * (t.v[2] * 1000007 + 3));
        }
    };
}

struct Tetrahedron
{
    Tetrahedron(int v0, int v1, int v2, int v3) :
        v{v0, v1, v2, v3}
    {
    }

    inline int t0v0() const {return v[0];}
    inline int t0v1() const {return v[1];}
    inline int t0v2() const {return v[2];}
    inline Triangle t0 () const
    {
        return Triangle(t0v0(), t0v1(), t0v2());
    }

    inline int t1v0() const {return v[0];}
    inline int t1v1() const {return v[2];}
    inline int t1v2() const {return v[3];}
    inline Triangle t1 () const
    {
        return Triangle(t1v0(), t1v1(), t1v2());
    }

    inline int t2v0() const {return v[0];}
    inline int t2v1() const {return v[3];}
    inline int t2v2() const {return v[1];}
    inline Triangle t2 () const
    {
        return Triangle(t2v0(), t2v1(), t2v2());
    }

    inline int t3v0() const {return v[3];}
    inline int t3v1() const {return v[2];}
    inline int t3v2() const {return v[1];}
    inline Triangle t3 () const
    {
        return Triangle(t3v0(), t3v1(), t3v2());
    }

    int v[4];

    // Algo flag
    bool flag;

    // Data cache
    double circumRadius2;
    glm::dvec3 circumCenter;
    glm::ivec3 cId;
};

struct GridCell
{
    std::vector<int> vertId;
    std::unordered_set<Tetrahedron*> tetra;
};

enum class EDir {BACK,  BACK_RIGHT,
                 RIGHT, FRONT_RIGHT,
                 FRONT, FRONT_LEFT,
                 LEFT,  BACK_LEFT,
                 STATIC};

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
            const std::vector<glm::dvec3>& boundingVertices,
            const std::vector<Tetrahedron>& boundingTetrahedras);

    void compileFacesAttributes(
            std::vector<glm::dvec3>& vertices,
            std::vector<glm::dvec3>& normals,
            std::vector<glm::dvec3>& triEdges,
            std::vector<double>& tetQualities);

    void insertVertices(const std::vector<glm::dvec3>& vertices);

    int externalVertCount;
    std::vector<Vertex> vert;
    std::list<Tetrahedron*> tetra;
    std::vector<std::vector<int>> neighbors;


    int qualityCount;
    double qualityMean;
    double qualityVar;


private:
    bool isExternalTetraHedron(Tetrahedron* tet);
    double tetrahedronQuality(Tetrahedron* tet);
    void pushTriangle(
            std::vector<glm::dvec3>& vertices,
            std::vector<glm::dvec3>& normals,
            std::vector<glm::dvec3>& triEdges,
            std::vector<double>& tetQualities,
            const glm::dvec3& A,
            const glm::dvec3& B,
            const glm::dvec3& C,
            const glm::dvec3& n,
            double quality);



    void initializeGrid(int idStart, int idEnd);
    void insertCell(const glm::ivec3& cId);
    void pullupTetrahedrons(const glm::ivec3& cId);
    void insertVertexGrid(const glm::ivec3& cId, int vId);
    Tetrahedron* findBaseTetrahedron(const glm::ivec3& cId, int vId);
    bool isBase(int vId, Tetrahedron* tet);
    void findDelaunayBall(int vId, Tetrahedron* base, std::unordered_set<Triangle>& ball);
    void remeshDelaunayBall(const glm::ivec3& cId, int vId, const std::unordered_set<Triangle>& ball);
    void insertTetrahedronGrid(const glm::ivec3& cId, int v0, int v1, int v2, int v3);
    void removeTetrahedronGrid(Tetrahedron* tet);
    void tearDownGrid();
    void compileAdjacencyLists();

    // Bounding polyhedron dimensions
    glm::dvec3 cMin;
    glm::dvec3 cMax;
    glm::dvec3 cDim;

    // Computing grid
    glm::ivec3 gridSize;
    std::vector<std::vector<std::vector<GridCell>>> grid;

    // Algorithms's main structure (keep allocated memory)
    std::vector<std::pair<glm::ivec3, EDir>> _baseQueue;
    std::vector<Tetrahedron*> _ballQueue;
    std::vector<Tetrahedron*> _ballPreserved;
    std::vector<Vertex*> _ballTouched;
};


#endif // GPUMESH_MESH
