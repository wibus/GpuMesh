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
    int v[2];

    inline GpuEdge() {}
    inline GpuEdge(const MeshEdge& e) : v{e.v[0], e.v[1]} {}
    inline operator MeshEdge() const { return MeshEdge(v[0], v[1]); }
    inline operator glm::ivec2() const { return glm::ivec2(v[0], v[1]); }
};

struct GpuTri
{
    int v[3];

    inline GpuTri() {}
    inline GpuTri(const MeshTri& t) : v{t.v[0], t.v[1], t.v[2]} {}
    inline operator MeshTri() const { return MeshTri(v[0], v[1], v[2]); }
    inline operator glm::ivec3() const { return glm::ivec3(v[0], v[1], v[2]); }
};

struct GpuTet
{
    int v[4];

    inline GpuTet() {}
    inline GpuTet(const MeshTet& t) : v{t.v[0], t.v[1], t.v[2], t.v[3]} {}
    inline operator MeshTet() const { return MeshTet(v[0], v[1], v[2], v[3]); }
    inline operator glm::ivec4() const { return glm::ivec4(v[0], v[1], v[2], v[3]); }
};

struct GpuPri
{
    int v[6];

    inline GpuPri() {}
    inline GpuPri(const MeshPri& p) : v{p.v[0], p.v[1], p.v[2], p.v[3], p.v[4], p.v[5]} {}
    inline operator MeshPri() const { return MeshPri(v[0], v[1], v[2], v[3], v[4], v[5]); }
};

struct GpuHex
{
    int v[8];

    inline GpuHex() {}
    inline GpuHex(const MeshHex& h) : v{h.v[0], h.v[1], h.v[2], h.v[3], h.v[4], h.v[5], h.v[6], h.v[7]} {}
    inline operator MeshHex() const { return MeshHex(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]); }
};

struct GpuNeigVert
{
    int v;

    inline GpuNeigVert() {}
    inline GpuNeigVert(const MeshNeigVert& n) : v(n.v) {}
    inline operator MeshNeigVert() const {return MeshNeigVert(v); }
};

struct GpuNeigElem
{
    int type;
    int id;

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
    int neigVertBase;

    // Neighbor vertices count
    int neigVertCount;

    // Neighbor elements list start location
    int neigElemBase;

    // Neighbor elements count
    int neigElemCount;

    inline GpuTopo() : type(0), neigVertBase(0), neigVertCount(0),
                                neigElemBase(0), neigElemCount(0)  {}
    inline GpuTopo(int type, int neigVertBase, int neigVertCount,
                             int neigElemBase, int neigElemCount) :
        type(type), neigVertBase(neigVertBase), neigVertCount(neigVertCount),
                    neigElemBase(neigElemBase), neigElemCount(neigElemCount) {}
};


class GpuMesh : public Mesh
{
public:
    GpuMesh();
    virtual ~GpuMesh();

    virtual void clear() override;

    virtual void compileTopoly() override;
    virtual void updateGpuTopoly() override;
    virtual void updateGpuVertices() override;
    virtual void updateCpuVertices() override;

    virtual std::string meshGeometryShaderName() const override;
    virtual void uploadGeometry(cellar::GlProgram& program) const override;
    virtual unsigned int glBuffer(const EMeshBuffer& buffer) const override;
    virtual void bindShaderStorageBuffers() const override;
    virtual size_t firstFreeBufferBinding() const override;

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
};




// IMPLEMENTATION //

#endif // GPUMESH_GPUMESH
