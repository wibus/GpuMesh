#ifndef GPUMESH_MESH
#define GPUMESH_MESH

#include <vector>
#include <functional>

#include <GLM/glm.hpp>


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
    int v[2];

    inline MeshEdge() : v{-1, -1} {}
    inline MeshEdge(int v0, int v1) : v{v0, v1} {}
    inline int& operator[] (int i) { return v[i]; }
    inline const int& operator[] (int i) const { return v[i]; }
};

struct MeshTri
{
    int v[3];

    inline MeshTri() : v{-1, -1, -1} {}
    inline MeshTri(int v0, int v1, int v2) : v{v0, v1, v2} {}
    inline int& operator[] (int i) { return v[i]; }
    inline const int& operator[] (int i) const { return v[i]; }
};


struct MeshTet
{
    int v[4];

    inline MeshTet() : v{-1, -1, -1, -1} {}
    inline MeshTet(int v0, int v1, int v2, int v3) : v{v0, v1, v2, v3} {}
    inline int& operator[] (int i) { return v[i]; }
    inline const int& operator[] (int i) const { return v[i]; }

    static const int EDGE_COUNT = 6;
    static const MeshEdge edges[EDGE_COUNT];
    static const int TRI_COUNT = 4;
    static const MeshTri tris[TRI_COUNT];
    static const int TET_COUNT = 1;
    static const MeshTet tets[TET_COUNT];
};


struct MeshPri
{
    int v[6];

    inline MeshPri() : v{-1, -1, -1, -1, -1, -1} {}
    inline MeshPri(int v0, int v1, int v2,
            int v3, int v4, int v5) :
        v{v0, v1, v2, v3, v4, v5} {}
    inline int operator[] (int i) { return v[i]; }
    inline const int& operator[] (int i) const { return v[i]; }

    static const int EDGE_COUNT = 9;
    static const MeshEdge edges[EDGE_COUNT];
    static const int TRI_COUNT = 8;
    static const MeshTri tris[TRI_COUNT];
    static const int TET_COUNT = 3;
    static const MeshTet tets[TET_COUNT];
};


struct MeshHex
{
    int v[8];

    inline MeshHex() : v{-1, -1, -1, -1, -1, -1, -1, -1} {}
    inline MeshHex(int v0, int v1, int v2, int v3,
            int v4, int v5, int v6, int v7) :
        v{v0, v1, v2, v3, v4, v5, v6, v7} {}
    inline int operator[] (int i) { return v[i]; }
    inline const int& operator[] (int i) const { return v[i]; }

    static const int EDGE_COUNT = 12;
    static const MeshEdge edges[EDGE_COUNT];
    static const int TRI_COUNT = 12;
    static const MeshTri tris[TRI_COUNT];
    static const int TET_COUNT = 5;
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

struct MeshTopo
{
    bool isFixed;
    std::vector<int> neighbors;

    bool isBoundary;
    const MeshBound& boundaryCallback;
    static const MeshBound NO_BOUNDARY;

    MeshTopo();
    MeshTopo(bool isFixed);
    MeshTopo(const MeshBound& boundaryCallback);
};


enum class EMeshBuffer
{
    VERT,

    TOPO,
    NEIG,

    TET,
    PRI,
    HEX,
};

namespace cellar
{
    class GlProgram;
}

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


    std::vector<MeshVert> vert;
    std::vector<MeshTopo> topo;
    std::vector<MeshTet> tetra;
    std::vector<MeshPri> prism;
    std::vector<MeshHex> hexa;


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
