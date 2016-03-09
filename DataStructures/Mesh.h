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

struct LocalTet
{
    inline LocalTet()
        { v[0] = -1; v[1] = -1; v[2] = -1; v[3] = -1;
          n[0] = -1; n[1] = -1; n[2] = -1; n[3] = -1;}

    inline LocalTet(uint v0, uint v1, uint v2, uint v3)
        { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
          n[0] = -1; n[1] = -1; n[2] = -1; n[3] = -1;}

    inline LocalTet(const MeshTet& t)
        { v[0] = t.v[0]; v[1] = t.v[1]; v[2] = t.v[2]; v[3] = t.v[3];
          n[0] = -1;     n[1] = -1;     n[2] = -1;     n[3] = -1;     }

    // Vertices of the tetrahedron
    uint v[4];

    // Neighbors of the tetrahedron
    //   n[0] is the neighbor tetrahedron
    //   at the oposite face of vertex v[0]
    uint n[4];
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
    REF_VERTS_BUFFER_BINDING,
    REF_METRICS_BUFFER_BINDING,
    KD_TETS_BUFFER_BINDING,
    KD_NODES_BUFFER_BINDING,
    LOCAL_TETS_BUFFER_BINDING,
    LOCAL_CACHE_BUFFER_BINDING
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

#endif // GPUMESH_MESH
