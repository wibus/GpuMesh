#ifndef GPUMESH_MESH
#define GPUMESH_MESH

#include <unordered_set>
#include <list>
#include <vector>
#include <memory>
#include <algorithm>

#include <GLM/glm.hpp>


struct Tetrahedron;

struct TetListNode
{
    Tetrahedron* tet;
    TetListNode* next;

    TetListNode() :
        tet(nullptr),
        next(nullptr)
    {
    }
};

struct TetList
{
    TetListNode* head;

    TetList() :
        head(nullptr)
    {
    }

    inline void addTet(Tetrahedron* tet)
    {
        TetListNode* node = _acquireNode();
        node->next = head;
        node->tet = tet;
        head = node;
    }

    inline void delTet(Tetrahedron* tet)
    {
        TetListNode* node = head;
        TetListNode* parent = nullptr;
        while(node != nullptr)
        {
            if(node->tet == tet)
            {
                if(parent != nullptr)
                    parent->next = node->next;
                else
                    head = node->next;
                _disposeNode(node);
                return;
            }

            parent = node;
            node = node->next;
        }
    }

    inline void clrTet()
    {
        TetListNode* node = head;
        while(node != nullptr)
        {
            TetListNode* next = node->next;
            delete node;
            node = next;
        }
        head = nullptr;
    }

    static void releaseMemoryPool()
    {
        int nodeCount = _nodePool.size();
        for(int i=0; i < nodeCount; ++i)
            delete _nodePool[i];

        _nodePool.clear();
        _nodePool.shrink_to_fit();
    }

private:
    inline static TetListNode* _acquireNode()
    {
        if(_nodePool.empty())
        {
            return new TetListNode();
        }
        else
        {
            TetListNode* node = _nodePool.back();
            _nodePool.pop_back();
            return node;
        }
    }

    inline static void _disposeNode(TetListNode* node)
    {
        _nodePool.push_back(node);
    }

    static std::vector<TetListNode*> _nodePool;
};

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
    int visitTime;

    // Data cache
    double circumRadius2;
    glm::dvec3 circumCenter;
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

const glm::ivec3 DIR[DIR_COUNT] = {
    glm::ivec3( 0,  0,  0),
    glm::ivec3(-1,  0,  0), glm::ivec3(-1, -1,  0),
    glm::ivec3( 0, -1,  0), glm::ivec3( 1, -1,  0),
    glm::ivec3( 1,  0,  0), glm::ivec3( 1,  1,  0),
    glm::ivec3( 0,  1,  0), glm::ivec3(-1,  1,  0),

    glm::ivec3( 0,  0, -1),
    glm::ivec3(-1,  0, -1), glm::ivec3(-1, -1, -1),
    glm::ivec3( 0, -1, -1), glm::ivec3( 1, -1, -1),
    glm::ivec3( 1,  0, -1), glm::ivec3( 1,  1, -1),
    glm::ivec3( 0,  1, -1), glm::ivec3(-1,  1, -1)
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
            const std::vector<glm::dvec3>& boundingVertices,
            const std::vector<Tetrahedron>& boundingTetrahedras);

    void compileAdjacencyLists();

    void compileFacesAttributes(
            const glm::dvec4& cutPlaneEq,
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
    void insertVertexGrid(const glm::ivec3& cId, int vId);
    Tetrahedron* findBaseTetrahedron(const glm::ivec3& cId, int vId);
    bool isBase(int vId, Tetrahedron* tet);
    void findDelaunayBall(int vId, Tetrahedron* base, std::unordered_set<Triangle>& ball);
    void remeshDelaunayBall(const glm::ivec3& cId, int vId, const std::unordered_set<Triangle>& ball);
    void insertTetrahedronGrid(int v0, int v1, int v2, int v3);
    void removeTetrahedronGrid(Tetrahedron* tet);
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
    int _currentVisitTime;

    // Memory pool
    inline static Tetrahedron* _acquireTetrahedron(int v0, int v1, int v2, int v3)
    {
        if(_tetPool.empty())
        {
            return new Tetrahedron(v0, v1, v2, v3);
        }
        else
        {
            Tetrahedron* tet = _tetPool.back();
            _tetPool.pop_back();
            tet->v[0] = v0;
            tet->v[1] = v1;
            tet->v[2] = v2;
            tet->v[3] = v3;
            return tet;
        }
    }

    inline static void _disposeTetrahedron(Tetrahedron* tet)
    {
        _tetPool.push_back(tet);
    }

    inline static void _releaseTetrahedronMemoryPool()
    {
        int tetCount = _tetPool.size();
        for(int i=0; i < tetCount; ++i)
            delete _tetPool[i];

        _tetPool.clear();
        _tetPool.shrink_to_fit();
    }

    static std::vector<Tetrahedron*> _tetPool;
};


#endif // GPUMESH_MESH
