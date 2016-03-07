#ifndef GPUMESH_MESH
#define GPUMESH_MESH

#include <mutex>
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
    inline MeshEdge(uint v0, uint v1) :
        v{v0, v1} {}

    inline uint& operator[] (uint i) { return v[i]; }
    inline const uint& operator[] (uint i) const { return v[i]; }
};

struct MeshTri
{
    uint v[3];


	inline MeshTri() : v{0, 0, 0} {}
	inline MeshTri(uint v0, uint v1, uint v2) : 
		v{v0, v1, v2} {}

    inline uint& operator[] (uint i) { return v[i]; }
    inline const uint& operator[] (uint i) const { return v[i]; }
};


struct MeshTet
{
    uint v[4];
    double value;


	inline MeshTet() : v{0, 0, 0, 0} {}
	inline MeshTet(uint v0, uint v1, uint v2, uint v3) : 
		v{v0, v1, v2, v3} {}

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
    double value;


	inline MeshPri() : v{0, 0, 0, 0, 0, 0} {}
	inline MeshPri(uint v0, uint v1, uint v2,
		           uint v3, uint v4, uint v5) :
		v{ v0, v1, v2, v3, v4, v5 } {}

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
    double value;


	inline MeshHex() : v{0, 0, 0, 0, 0, 0, 0, 0} {}
	inline MeshHex(uint v0, uint v1, uint v2, uint v3,
		           uint v4, uint v5, uint v6, uint v7) :
		v{ v0, v1, v2, v3, v4, v5, v6, v7 } {}

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
    std::vector<MeshNeigVert> neighborVerts;
    std::vector<MeshNeigElem> neighborElems;

    bool isFixed;
    bool isBoundary;
    const MeshBound* snapToBoundary;
    static const MeshBound NO_BOUNDARY;

    MeshTopo();
    MeshTopo(bool isFixed);
    MeshTopo(const MeshBound* snapToBoundary);
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

    GROUP_MEMBERS
};

enum class EBufferBinding
{
    EVALUATE_QUALS_BUFFER_BINDING,
    VERTEX_ACCUMS_BUFFER_BINDING,
    KD_NODES_BUFFER_BINDING,
    KD_TETS_BUFFER_BINDING,
    KD_METRICS_BUFFER_BINDING
};

enum class ECutType
{
    None,
    VirtualPlane,
    PhysicalPlane,
    InvertedElements
};

class OptimizationPlot;

typedef void (*ModelBoundsCudaFct)(void);


class Mesh
{
public:
    Mesh();
    virtual ~Mesh();

    virtual void clear();

    virtual void compileTopology();
    virtual void updateGpuTopology();
    virtual void updateVerticesFromCpu();
    virtual void updateVerticesFromGlsl();
    virtual void updateVerticesFromCuda();

    virtual std::string meshGeometryShaderName() const;
    virtual void uploadGeometry(cellar::GlProgram& program) const;
    virtual unsigned int glBuffer(const EMeshBuffer& buffer) const;
    virtual unsigned int bufferBinding(EBufferBinding binding) const;
    virtual void bindShaderStorageBuffers() const;

    virtual std::string modelBoundsShaderName() const;
    virtual void setModelBoundsShaderName(const std::string& name);

    virtual ModelBoundsCudaFct modelBoundsCudaFct() const;
    virtual void setModelBoundsCudaFct(ModelBoundsCudaFct fct);

    virtual void printPropperties(OptimizationPlot& plot) const;

    void getVerts(glm::dvec3 vp[4], const MeshTet& tet) const;
    void getVerts(glm::dvec3 vp[6], const MeshPri& pri) const;
    void getVerts(glm::dvec3 vp[8], const MeshHex& hex) const;


    std::string modelName;
    std::vector<MeshVert> verts;
    std::vector<MeshTet>  tets;
    std::vector<MeshPri>  pris;
    std::vector<MeshHex>  hexs;
    std::vector<MeshTopo> topos;

    std::vector<std::vector<uint>> independentGroups;


protected:
    virtual void compileNeighborhoods();
    virtual void addEdge(int firstVert,
                         int secondVert);


    /// @brief Independent vertex groups compilation
    /// Compiles independent vertex groups that is used by parallel smoothing
    /// algorithms to ensure that no two _adjacent_ vertices are moved at the
    /// same time. Independent vertices are vertices that do not share a common
    /// element. This is more strict than prohibiting edge existance.
    ///
    /// A simple graph coloring scheme is used to generate the groups. The
    /// algorithm works well with connected and highly disconnected graphs and
    /// show a linear complexity in either case : O(n*d), where _n_ is the
    /// number of vertices and _d_ is the 'mean' vertex degree.
    virtual void compileIndependentGroups();

    std::string _modelBoundsShaderName;
    ModelBoundsCudaFct _modelBoundsCudaFct;
};



// IMPLEMENTATION //
inline void Mesh::getVerts(glm::dvec3 vp[4], const MeshTet& tet) const
{
    vp[0] = verts[tet.v[0]].p;
    vp[1] = verts[tet.v[1]].p;
    vp[2] = verts[tet.v[2]].p;
    vp[3] = verts[tet.v[3]].p;
}

inline void Mesh::getVerts(glm::dvec3 vp[6], const MeshPri& pri) const
{
    vp[0] = verts[pri.v[0]].p;
    vp[1] = verts[pri.v[1]].p;
    vp[2] = verts[pri.v[2]].p;
    vp[3] = verts[pri.v[3]].p;
    vp[4] = verts[pri.v[4]].p;
    vp[5] = verts[pri.v[5]].p;
}

inline void Mesh::getVerts(glm::dvec3 vp[8], const MeshHex& hex) const
{
    vp[0] = verts[hex.v[0]].p;
    vp[1] = verts[hex.v[1]].p;
    vp[2] = verts[hex.v[2]].p;
    vp[3] = verts[hex.v[3]].p;
    vp[4] = verts[hex.v[4]].p;
    vp[5] = verts[hex.v[5]].p;
    vp[6] = verts[hex.v[6]].p;
    vp[7] = verts[hex.v[7]].p;
}

#endif // GPUMESH_MESH
