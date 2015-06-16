#ifndef GPUMESH_MESH
#define GPUMESH_MESH

#include <vector>

#include <GLM/glm.hpp>

#include "Tetrahedron.h"
#include "TetList.h"
#include "TetPool.h"
#include "TriSet.h"


struct Vertex
{
    Vertex() {}
    Vertex(const glm::dvec3 pos) :
        p(pos),
        isBoundary(false)
    {}

    glm::dvec3 p;
    TetList tetList;
    bool isBoundary;

    // Algo flag
    int visitTime;
};


struct GridCell
{
    std::vector<int> insertedVertId;
    std::vector<int> waitingVertId;
};


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


class Mesh
{
public:

    unsigned int vertCount() const;
    unsigned int elemCount() const;

    void initialize(
            const std::vector<glm::dvec3>& boundingVertices,
            const std::vector<Tetrahedron>& boundingTetrahedras);

    void compileAdjacencyLists();
    void compileTetrahedronQuality(
            double& qualityMean,
            double& qualityVar);

    void compileFacesAttributes(
            const glm::dvec4& cutPlaneEq,
            std::vector<glm::vec3>& vertices,
            std::vector<glm::vec3>& normals,
            std::vector<glm::vec3>& triEdges,
            std::vector<float>& colors);


    void insertVertices(const std::vector<glm::dvec3>& vertices);

    int externalVertCount;
    std::vector<Vertex> vert;
    std::vector<Tetrahedron*> tetra;
    std::vector<std::vector<int>> neighbors;


private:
    bool isExternalTetraHedron(Tetrahedron* tet);
    double tetrahedronQuality(Tetrahedron* tet);
    void pushTriangle(
            std::vector<glm::vec3>& vertices,
            std::vector<glm::vec3>& normals,
            std::vector<glm::vec3>& triEdges,
            std::vector<float>& tetQualities,
            const glm::dvec3& A,
            const glm::dvec3& B,
            const glm::dvec3& C,
            const glm::dvec3& n,
            double quality);

    inline bool intersects(const glm::dvec3& v, Tetrahedron* tet);

    void initializeGrid(int idStart, int idEnd);
    void insertCell(const glm::ivec3& cId);
    void insertVertexGrid(const glm::ivec3& cId, int vId);
    Tetrahedron* findBaseTetrahedron(const glm::ivec3& cId, int vId);
    void findDelaunayBall(const glm::ivec3& cId, int vId);
    void remeshDelaunayBall(int vId);
    void insertTetrahedronGrid(int v0, int v1, int v2, int v3);
    void removeTetrahedronGrid(Tetrahedron* tet);
    void makeTetrahedronPositive(Tetrahedron* tet);
    void tearDownGrid();

    // Bounding polyhedron dimensions
    glm::dvec3 cMin;
    glm::dvec3 cMax;
    glm::dvec3 cExtMin;
    glm::dvec3 cExtMax;
    glm::dvec3 cExtDim;

    // Computing grid
    glm::ivec3 gridSize;
    std::vector<std::vector<std::vector<GridCell>>> grid;

    // Algorithms's main structure (keep allocated memory)
    std::vector<std::pair<glm::ivec3, EDir>> _baseQueue;
    std::vector<Vertex*> _ballQueue;
    TetPool _tetPool;
    TriSet _ball;

    int _currentVisitTime;
};


#endif // GPUMESH_MESH
