#ifndef GPUMESH_CPUDELAUNAYMESHER
#define GPUMESH_CPUDELAUNAYMESHER

#include <memory>

#include "AbstractMesher.h"

#include "DataStructures/Tetrahedron.h"
#include "DataStructures/TetPool.h"
#include "DataStructures/TriSet.h"


struct Vertex;
struct GridCell;

enum EDir {
    STATIC,
    BACK,       BACK_RIGHT,
    RIGHT,      FRONT_RIGHT,
    FRONT,      FRONT_LEFT,
    LEFT,       BACK_LEFT,

    DOWN,
    BACK_DOWN,  BACK_RIGHT_DOWN,
    RIGHT_DOWN, FRONT_RIGHT_DOWN,
    FRONT_DOWN, FRONT_LEFT_DOWN,
    LEFT_DOWN,  BACK_LEFT_DOWN,

    DIR_COUNT
};


class CpuDelaunayMesher : public AbstractMesher
{
public:
    CpuDelaunayMesher();
    virtual ~CpuDelaunayMesher();

protected:
    virtual void genBox(Mesh& mesh, size_t vertexCount);
    virtual void genShell(Mesh& mesh, size_t vertexCount);
    virtual void genSphere(Mesh& mesh, size_t vertexCount);

    virtual void insertBoundingMesh();
    virtual void insertVertices(Mesh& mesh, const std::vector<glm::dvec3>& vertices);

    void initializeGrid(int idStart, int idEnd);
    void insertCell(const glm::ivec3& cId);
    void insertVertexGrid(const glm::ivec3& cId, int vId);
    Tetrahedron* findBaseTetrahedron(const glm::ivec3& cId, int vId);
    void findDelaunayBall(const glm::ivec3& cId, int vId);
    void remeshDelaunayBall(int vId);
    void insertTetrahedronGrid(int v0, int v1, int v2, int v3);
    void removeTetrahedronGrid(Tetrahedron* tet);
    void tearDownGrid(Mesh& mesh);

    inline bool intersects(const glm::dvec3& v, Tetrahedron* tet);
    bool isExternalTetraHedron(Tetrahedron* tet);
    void makeTetrahedronPositive(Tetrahedron* tet);

private:
    // Boundaries
    std::unique_ptr<MeshBound> _boxXFaceBoundary;
    std::unique_ptr<MeshBound> _boxYFaceBoundary;
    std::unique_ptr<MeshBound> _boxZFaceBoundary;
    std::unique_ptr<MeshBound> _sphereInBoundary;
    std::unique_ptr<MeshBound> _sphereOutBoundary;

    // Main data structures
    std::vector<Vertex> vert;
    std::vector<Tetrahedron*> tetra;

    // Bounding polyhedron dimensions
    glm::dvec3 cMin;
    glm::dvec3 cMax;
    glm::dvec3 cExtMin;
    glm::dvec3 cExtMax;
    glm::dvec3 cExtDim;

    // Computing grid
    glm::ivec3 gridSize;
    std::vector<std::vector<std::vector<GridCell>>> grid;

    // Algorithms's data structures (keep allocated memory)
    std::vector<std::pair<glm::ivec3, EDir>> _baseQueue;
    std::vector<Vertex*> _ballQueue;
    TetPool _tetPool;
    TriSet _ball;

    int _currentVisitTime;
    int _externalVertCount;
};

#endif // GPUMESH_CPUDELAUNAYMESHER
