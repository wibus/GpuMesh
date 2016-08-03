#ifndef GPUMESH_MESH
#define GPUMESH_MESH

#include <mutex>
#include <vector>
#include <memory>
#include <functional>

#include <GLM/glm.hpp>

#ifndef uint
typedef unsigned int uint;
#endif // uint

namespace cellar
{
    class GlProgram;
}


class AbstractBoundary;
class AbstractConstraint;
class OptimizationPlot;
class NodeGroups;


struct MeshVert
{
    glm::dvec3 p;
    mutable uint c;


    inline MeshVert() : p(0) {}
    inline MeshVert(const glm::dvec3 p) : p(p), c(0){}
    inline MeshVert(const glm::dvec3 p, uint c) : p(p), c(c){}
    inline double& operator[] (int d) { return p[d]; }
    inline const double& operator[] (int d) const { return p[d]; }
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
    mutable uint c[1];


    inline MeshTet() :
        v{0, 0, 0, 0},
        c{0}
    {}
	inline MeshTet(uint v0, uint v1, uint v2, uint v3) : 
        v{v0, v1, v2, v3},
        c{0}
    {}
    inline MeshTet(uint v0, uint v1, uint v2, uint v3, uint c) :
        v{v0, v1, v2, v3},
        c{c}
    {}

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

struct MeshLocalTet
{
    inline MeshLocalTet()
        { v[0] = -1; v[1] = -1; v[2] = -1; v[3] = -1;
          n[0] = -1; n[1] = -1; n[2] = -1; n[3] = -1;}

    inline MeshLocalTet(uint v0, uint v1, uint v2, uint v3)
        { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
          n[0] = -1; n[1] = -1; n[2] = -1; n[3] = -1;}

    inline MeshLocalTet(const MeshTet& t)
        { v[0] = t.v[0]; v[1] = t.v[1]; v[2] = t.v[2]; v[3] = t.v[3];
          n[0] = -1;     n[1] = -1;     n[2] = -1;     n[3] = -1;     }

    operator MeshTet () const {return MeshTet(v[0], v[1], v[2], v[3]);}

    // Vertices of the tetrahedron
    uint v[4];

    // Neighbors of the tetrahedron
    //   n[0] is the neighbor tetrahedron
    //   at the oposite face of vertex v[0]
    uint n[4];
};

struct MeshPri
{
    uint v[6];
    double value;
    mutable uint c[6];


    inline MeshPri() :
        v{0, 0, 0, 0, 0, 0},
        c{0, 0, 0, 0, 0, 0}
    {}
	inline MeshPri(uint v0, uint v1, uint v2,
                   uint v3, uint v4, uint v5) :
        v{v0, v1, v2, v3, v4, v5},
        c{0, 0, 0, 0, 0, 0}
    {}
    inline MeshPri(uint v0, uint v1, uint v2,
                   uint v3, uint v4, uint v5,
                   uint c0, uint c1, uint c2,
                   uint c3, uint c4, uint c5) :
        v{v0, v1, v2, v3, v4, v5},
        c{c0, c1, c2, c3, c4, c5}
    {}

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
    mutable uint c[8];


    inline MeshHex() :
        v{0, 0, 0, 0, 0, 0, 0, 0},
        c{0, 0, 0, 0, 0, 0, 0, 0}
    {}
	inline MeshHex(uint v0, uint v1, uint v2, uint v3,
		           uint v4, uint v5, uint v6, uint v7) :
        v{v0, v1, v2, v3, v4, v5, v6, v7},
        c{0, 0, 0, 0, 0, 0, 0, 0}
    {}
    inline MeshHex(uint v0, uint v1, uint v2, uint v3,
                   uint v4, uint v5, uint v6, uint v7,
                   uint c0, uint c1, uint c2, uint c3,
                   uint c4, uint c5, uint c6, uint c7) :
        v{v0, v1, v2, v3, v4, v5, v6, v7},
        c{c0, c1, c2, c3, c4, c5, c6, c7}
    {}

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
    inline operator uint() const {return id;}
};

struct MeshTopo
{
    std::vector<MeshNeigVert> neighborVerts;
    std::vector<MeshNeigElem> neighborElems;

    const AbstractConstraint* snapToBoundary;
    static const AbstractConstraint* NO_BOUNDARY;

    MeshTopo();
    explicit MeshTopo(const glm::dvec3& fixedPosition);
    explicit MeshTopo(const AbstractConstraint* constraint);
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
    EVALUATE_QUAL_BUFFER_BINDING,
    EVALUATE_HIST_BUFFER_BINDING,
    VERTEX_ACCUMS_BUFFER_BINDING,
    REF_VERTS_BUFFER_BINDING,
    REF_METRICS_BUFFER_BINDING,
    KD_TETS_BUFFER_BINDING,
    KD_NODES_BUFFER_BINDING,
    LOCAL_TETS_BUFFER_BINDING,
    SPAWN_OFFSETS_BUFFER_BINDING
};

enum class ECutType
{
    None,
    VirtualPlane,
    PhysicalPlane,
    InvertedElements
};


class Mesh
{
public:
    Mesh();
    virtual ~Mesh();

    virtual Mesh& operator=(const Mesh& mesh);

    virtual void clear();

    virtual void compileTopology(bool verbose = true);

    virtual void updateGlslTopology() const;
    virtual void updateGlslVertices() const;
    virtual void fetchGlslVertices();
    virtual void clearGlslMemory() const;

    virtual void updateCudaTopology() const;
    virtual void updateCudaVertices() const;
    virtual void fetchCudaVertices();
    virtual void clearCudaMemory() const;

    virtual std::string meshGeometryShaderName() const;
    virtual unsigned int glBuffer(const EMeshBuffer& buffer) const;
    virtual unsigned int glBufferBinding(EBufferBinding binding) const;
    virtual void bindGlShaderStorageBuffers() const;

    virtual void printPropperties(OptimizationPlot& plot) const;


    NodeGroups& nodeGroups() const;

    AbstractBoundary& boundary() const;
    void setBoundary(const std::shared_ptr<AbstractBoundary>& boundary);


    std::string modelName;
    std::vector<MeshVert> verts;
    std::vector<MeshTopo> topos;
    std::vector<MeshTet>  tets;
    std::vector<MeshPri>  pris;
    std::vector<MeshHex>  hexs;


protected:
    virtual void compileNeighborhoods();
    virtual void addEdge(int firstVert,
                         int secondVert);

    std::shared_ptr<NodeGroups> _nodeGroups;
    std::shared_ptr<AbstractBoundary> _boundary;
};



// IMPLEMENTATION //
inline NodeGroups& Mesh::nodeGroups() const
{
    return *_nodeGroups;
}

inline AbstractBoundary& Mesh::boundary() const
{
    return *_boundary;
}

#endif // GPUMESH_MESH
