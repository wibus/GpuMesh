#ifndef GPUMESH_GPUMESH
#define GPUMESH_GPUMESH

#include <climits>

#include <GL3/gl3w.h>

#include "Mesh.h"


struct GpuVert
{
    glm::vec3 p;
    GLuint c;

    inline GpuVert() {}
    inline GpuVert(const MeshVert& v) : p(v.p), c(v.c) {}
    inline operator MeshVert() const { return MeshVert(p, c); }
};

struct GpuEdge
{
    GLuint v[2];

    inline GpuEdge() {}
    inline GpuEdge(const MeshEdge& e) : v{ e.v[0], e.v[1] } {}
    inline operator MeshEdge() const { return MeshEdge(v[0], v[1]); }
    inline operator glm::ivec2() const { return glm::ivec2(v[0], v[1]); }
};

struct GpuTri
{
    GLuint v[3];

    inline GpuTri() {}
    inline GpuTri(const MeshTri& t) : v{t.v[0], t.v[1], t.v[2]} {}
    inline operator MeshTri() const { return MeshTri(v[0], v[1], v[2]); }
    inline operator glm::ivec3() const { return glm::ivec3(v[0], v[1], v[2]); }
};

struct GpuTet
{
    GLuint v[4];
    GLuint c[1];

    inline GpuTet() {}
    inline GpuTet(const MeshTet& t) :
        v{t.v[0], t.v[1], t.v[2], t.v[3]},
        c{t.c[0]}
    {}
    inline operator MeshTet() const { return MeshTet(v[0], v[1], v[2], v[3], c[0]); }
};

struct GpuPri
{
    GLuint v[6];
    GLuint c[6];

    inline GpuPri() {}
    inline GpuPri(const MeshPri& p) :
        v{p.v[0], p.v[1], p.v[2], p.v[3], p.v[4], p.v[5]},
        c{p.c[0], p.c[1], p.c[2], p.c[3], p.c[4], p.c[5]}
    {}
    inline operator MeshPri() const
    {
        return MeshPri(v[0], v[1], v[2], v[3], v[4], v[5],
                       c[0], c[1], c[2], c[3], c[4], c[5]);
    }
};

struct GpuHex
{
    GLuint v[8];
    GLuint c[8];

    inline GpuHex() {}
    inline GpuHex(const MeshHex& h) :
        v{h.v[0], h.v[1], h.v[2], h.v[3], h.v[4], h.v[5], h.v[6], h.v[7]},
        c{h.c[0], h.c[1], h.c[2], h.c[3], h.c[4], h.c[5], h.c[6], h.c[7]}
    {}
    inline operator MeshHex() const
    {
        return MeshHex(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                       c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
    }
};

struct GpuNeigVert
{
    GLuint v;

    inline GpuNeigVert() {}
    inline GpuNeigVert(const MeshNeigVert& n) : v(n.v) {}
    inline operator MeshNeigVert() const {return MeshNeigVert(v); }
};

struct GpuNeigElem
{
    int type;
    GLuint id;

    inline GpuNeigElem() {}
    inline GpuNeigElem(const MeshNeigElem& n) : type(n.type), id(n.id) {}
    inline operator MeshNeigElem() const {return MeshNeigElem(type, id); }
};

struct GpuTopo
{
    // Type of vertex :
    //  * -1 = free
    //  *  0 = fixed
    //  * >0 = boundary
    int type;

    // Neighbor vertices list start location
    GLuint neigVertBase;

    // Neighbor vertices count
    GLuint neigVertCount;

    // Neighbor elements list start location
    GLuint neigElemBase;

    // Neighbor elements count
    GLuint neigElemCount;

    inline GpuTopo() :
        type(0),
        neigVertBase(0), neigVertCount(0),
        neigElemBase(0), neigElemCount(0)  {}
    inline GpuTopo(int type,
                   uint neigVertBase, uint neigVertCount,
                   uint neigElemBase, uint neigElemCount) :
        type(type),
        neigVertBase(neigVertBase), neigVertCount(neigVertCount),
        neigElemBase(neigElemBase), neigElemCount(neigElemCount) {}
};

struct GpuKdNode
{
    GpuKdNode() :
        left(-1),
        right(-1),
        tetBeg(0),
        tetEnd(0)
    {}

    GLint left;
    GLint right;

    GLuint tetBeg;
    GLuint tetEnd;

    glm::vec4 separator;
};

struct GpuLocalTet
{
    GpuLocalTet() :
        v{0, 0, 0, 0},
        n{0, 0, 0, 0}
    {}

    GpuLocalTet(const uint v[4], const uint n[4]) :
        v{v[0], v[1], v[2], v[3]},
        n{n[0], n[1], n[2], n[3]}
    {}

    GLuint v[4];
    GLuint n[4];
};


class GpuMesh : public Mesh
{
public:
    GpuMesh();
    virtual ~GpuMesh();

    virtual void clear() override;

    virtual void updateGlslTopology() const override;
    virtual void updateGlslVertices() const override;
    virtual void fetchGlslVertices() override;
    virtual void clearGlslMemory() const override;

    virtual void updateCudaTopology() const override;
    virtual void updateCudaVertices() const override;
    virtual void fetchCudaVertices() override;
    virtual void clearCudaMemory() const override;


    virtual std::string meshGeometryShaderName() const override;
    virtual unsigned int glBuffer(const EMeshBuffer& buffer) const override;
    virtual unsigned int glBufferBinding(EBufferBinding binding) const override;
    virtual void bindGlShaderStorageBuffers() const override;


protected:
    void buildGpuTetBuffer(
        std::vector<GpuTet>& tetBuff) const;

    void buildGpuPriBuffer(
        std::vector<GpuPri>& priBuff) const;

    void buildGpuHexBuffer(
        std::vector<GpuHex>& hexBuff) const;

    void buildGpuTopoBuffers(
        std::vector<GpuTopo>& topoBuff,
        std::vector<GpuNeigVert>& neigVertBuff,
        std::vector<GpuNeigElem>& neigElemBuff) const;


    // SSBO indices are mutable because they are allocated
    // by update<Impl>{Topology, Vertices}() methods.

    // These methods were made const because they are needed to push
    // the mesh on the GPU before performing a simple mesh evaluation.

    mutable GLuint _vertSsbo;
    mutable GLuint _tetSsbo;
    mutable GLuint _priSsbo;
    mutable GLuint _hexSsbo;
    mutable GLuint _topoSsbo;
    mutable GLuint _neigVertSsbo;
    mutable GLuint _neigElemSsbo;
    mutable GLuint _groupMembersSsbo;
};

#endif // GPUMESH_GPUMESH
