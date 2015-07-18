#ifndef GPUMESH_MESH
#define GPUMESH_MESH

#include <vector>
#include <functional>

#include <GLM/glm.hpp>

#ifndef uint
typedef unsigned int uint;
#endif // uint

namespace cellar
{
    class GlProgram;
}

struct MeshVert
{
    glm::dvec3 p;

    inline MeshVert() : p(0) {}
    inline MeshVert(const glm::dvec3 p) : p(p){}
    inline double& operator[] (int c) { return p[c]; }
    inline const double& operator[] (int c) const { return p[c]; }
    inline operator glm::dvec3() const { return p; }
};

struct MeshEdge
{
    uint v[2];

    inline MeshEdge() : v{0, 0} {}
    inline MeshEdge(uint v0, uint v1) : v{v0, v1} {}
    inline uint& operator[] (uint i) { return v[i]; }
    inline const uint& operator[] (uint i) const { return v[i]; }
};

struct MeshTri
{
    uint v[3];

    inline MeshTri() : v{0, 0, 0} {}
    inline MeshTri(uint v0, uint v1, uint v2) : v{v0, v1, v2} {}
    inline uint& operator[] (uint i) { return v[i]; }
    inline const uint& operator[] (uint i) const { return v[i]; }
};


struct MeshTet
{
    uint v[4];

    inline MeshTet() : v{0, 0, 0, 0} {}
    inline MeshTet(uint v0, uint v1, uint v2, uint v3) : v{v0, v1, v2, v3} {}
    inline uint& operator[] (uint i) { return v[i]; }
    inline const uint& operator[] (uint i) const { return v[i]; }

    static const int ELEMENT_TYPE = 0;
    static const uint VERTEX_COUNT = 4;
    static const uint EDGE_COUNT = 6;
    static const MeshEdge edges[EDGE_COUNT];
    static const uint TRI_COUNT = 4;
    static const MeshTri tris[TRI_COUNT];
    static const uint TET_COUNT = 1;
    static const MeshTet tets[TET_COUNT];
};


struct MeshPri
{
    uint v[6];

    inline MeshPri() : v{0, 0, 0, 0, 0, 0} {}
    inline MeshPri(uint v0, uint v1, uint v2,
                   uint v3, uint v4, uint v5) :
        v{v0, v1, v2, v3, v4, v5} {}
    inline uint operator[] (uint i) { return v[i]; }
    inline const uint& operator[] (uint i) const { return v[i]; }

    static const int ELEMENT_TYPE = 1;
    static const uint VERTEX_COUNT = 6;
    static const uint EDGE_COUNT = 9;
    static const MeshEdge edges[EDGE_COUNT];
    static const uint TRI_COUNT = 8;
    static const MeshTri tris[TRI_COUNT];
    static const uint TET_COUNT = 3;
    static const MeshTet tets[TET_COUNT];
};


struct MeshHex
{
    uint v[8];

    inline MeshHex() : v{0, 0, 0, 0, 0, 0, 0, 0} {}
    inline MeshHex(uint v0, uint v1, uint v2, uint v3,
                   uint v4, uint v5, uint v6, uint v7) :
        v{v0, v1, v2, v3, v4, v5, v6, v7} {}
    inline uint operator[] (uint i) { return v[i]; }
    inline const uint& operator[] (uint i) const { return v[i]; }

    static const int ELEMENT_TYPE = 2;
    static const uint VERTEX_COUNT = 8;
    static const uint EDGE_COUNT = 12;
    static const MeshEdge edges[EDGE_COUNT];
    static const uint TRI_COUNT = 12;
    static const MeshTri tris[TRI_COUNT];
    static const uint TET_COUNT = 5;
    static const MeshTet tets[TET_COUNT];
};

class MeshBound
{
public:
    MeshBound(int id);
    virtual ~MeshBound();

    inline int id() const {return _id;}

    virtual glm::dvec3 operator()(const glm::dvec3& pos) const;

private:
    int _id;
};

struct MeshNeigVert
{
    uint v;

    inline MeshNeigVert() : v(0) {}
    inline MeshNeigVert(uint v) : v(v) {}
    inline operator uint() const {return v;}
};

struct MeshNeigElem
{
    int type;
    uint id;

    inline MeshNeigElem() : type(-1), id(0) {}
    inline MeshNeigElem(int type, uint id) : type(type), id(id) {}
};

struct MeshTopo
{
    bool isFixed;
    std::vector<MeshNeigVert> neighborVerts;
    std::vector<MeshNeigElem> neighborElems;

    bool isBoundary;
    const MeshBound& snapToBoundary;
    static const MeshBound NO_BOUNDARY;

    MeshTopo();
    MeshTopo(bool isFixed);
    MeshTopo(const MeshBound& snapToBoundary);
};


enum class EMeshBuffer
{
    VERT,

    TET,
    PRI,
    HEX,

    TOPO,
    NEIG_VERT,
    NEIG_ELEM,
};


class Mesh
{
public:
    Mesh();
    virtual ~Mesh();

    unsigned int vertCount() const;
    unsigned int elemCount() const;

    virtual void clear();

    virtual void compileTopoly();
    virtual void updateGpuTopoly();
    virtual void updateGpuVertices();
    virtual void updateCpuVertices();

    virtual std::string meshGeometryShaderName() const;
    virtual void uploadGeometry(cellar::GlProgram& program) const;
    virtual unsigned int glBuffer(const EMeshBuffer& buffer) const;
    virtual void bindShaderStorageBuffers() const;
    virtual size_t firstFreeBufferBinding() const;


    std::vector<MeshVert> vert;
    std::vector<MeshTet>  tetra;
    std::vector<MeshPri>  prism;
    std::vector<MeshHex>  hexa;
    std::vector<MeshTopo> topo;


protected:
    virtual void addEdge(int firstVert,
                         int secondVert);
};




// IMPLEMENTATION //
inline unsigned int Mesh::vertCount() const
{
    return vert.size();
}

inline unsigned int Mesh::elemCount() const
{
    return tetra.size() + prism.size() + hexa.size();
}

#endif // GPUMESH_MESH
