#ifndef GPUMESH_GPUMESH
#define GPUMESH_GPUMESH

#include <climits>

#include <GL3/gl3w.h>

#include "Mesh.h"


struct GpuVert
{
    glm::vec4 p;

    inline GpuVert() {}
    inline GpuVert(const MeshVert& v) : p(v.p, 0.0) {}
    inline operator MeshVert() const { return glm::dvec3(p); }
};

struct GpuEdge
{
    GLuint v[2];

	inline GpuEdge() {}
#if defined(_MSC_VER) && _MSC_VER <= 1800
	inline GpuEdge(const MeshEdge& e) 
		{v[0] = e.v[0]; v[1] = e.v[1]; }
#else
	inline GpuEdge(const MeshEdge& e) : v{ e.v[0], e.v[1] } {}
#endif
    inline operator MeshEdge() const { return MeshEdge(v[0], v[1]); }
    inline operator glm::ivec2() const { return glm::ivec2(v[0], v[1]); }
};

struct GpuTri
{
    GLuint v[3];

	inline GpuTri() {}
#if defined(_MSC_VER) && _MSC_VER <= 1800
	inline GpuTri(const MeshTri& t)
		{v[0] = t.v[0]; v[1] = t.v[1]; v[2] = t.v[2];}
#else
	inline GpuTri(const MeshTri& t) : v{t.v[0], t.v[1], t.v[2]} {}
#endif
    inline operator MeshTri() const { return MeshTri(v[0], v[1], v[2]); }
    inline operator glm::ivec3() const { return glm::ivec3(v[0], v[1], v[2]); }
};

struct GpuTet
{
    GLuint v[4];

	inline GpuTet() {}
#if defined(_MSC_VER) && _MSC_VER <= 1800
	inline GpuTet(const MeshTet& t)
		{v[0] = t.v[0]; v[1] = t.v[1]; v[2] = t.v[2]; v[3] = t.v[3];}
#else
	inline GpuTet(const MeshTet& t) : v{t.v[0], t.v[1], t.v[2], t.v[3]} {}
#endif
    inline operator MeshTet() const { return MeshTet(v[0], v[1], v[2], v[3]); }
    inline operator glm::ivec4() const { return glm::ivec4(v[0], v[1], v[2], v[3]); }
};

struct GpuPri
{
    GLuint v[6];

	inline GpuPri() {}
#if defined(_MSC_VER) && _MSC_VER <= 1800
	inline GpuPri(const MeshPri& p)
		{v[0] = p.v[0]; v[1] = p.v[1]; v[2] = p.v[2]; 
		 v[3] = p.v[3]; v[4] = p.v[4]; v[5] = p.v[5];}
#else
	inline GpuPri(const MeshPri& p) : v{p.v[0], p.v[1], p.v[2], p.v[3], p.v[4], p.v[5]} {}
#endif
    inline operator MeshPri() const { return MeshPri(v[0], v[1], v[2], v[3], v[4], v[5]); }
};

struct GpuHex
{
    GLuint v[8];

	inline GpuHex() {}
#if defined(_MSC_VER) && _MSC_VER <= 1800
	inline GpuHex(const MeshHex& h)
		{v[0] = h.v[0]; v[1] = h.v[1]; v[2] = h.v[2]; v[3] = h.v[3]; 
		 v[4] = h.v[4]; v[5] = h.v[5]; v[6] = h.v[6]; v[7] = h.v[7];}
#else
	inline GpuHex(const MeshHex& h) : v{h.v[0], h.v[1], h.v[2], h.v[3], h.v[4], h.v[5], h.v[6], h.v[7]} {}
#endif
    inline operator MeshHex() const { return MeshHex(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]); }
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

    virtual void compileTopology() override;
    virtual void updateGpuTopology() override;
    virtual void updateVerticesFromCpu() override;
    virtual void updateVerticesFromGlsl() override;
    virtual void updateVerticesFromCuda() override;

    virtual std::string meshGeometryShaderName() const override;
    virtual void uploadGeometry(cellar::GlProgram& program) const override;
    virtual unsigned int glBuffer(const EMeshBuffer& buffer) const override;
    virtual unsigned int bufferBinding(EBufferBinding binding) const override;
    virtual void bindShaderStorageBuffers() const override;


protected:
    virtual void uploadElement(
            cellar::GlProgram& program,
            const std::string& prefix,
            int edgeCount, const MeshEdge edges[],
            int triCount,  const MeshTri tris[],
            int tetCount,  const MeshTet tets[]) const;

    GLuint _vertSsbo;
    GLuint _tetSsbo;
    GLuint _priSsbo;
    GLuint _hexSsbo;
    GLuint _topoSsbo;
    GLuint _neigVertSsbo;
    GLuint _neigElemSsbo;
    GLuint _groupMembersSsbo;
};




// IMPLEMENTATION //

#endif // GPUMESH_GPUMESH
