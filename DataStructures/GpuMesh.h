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

struct GpuNeig
{
    int v;

    inline GpuNeig() {}
    inline GpuNeig(int v) : v(v) {}
    inline operator int() const {return v; }
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

struct GpuTopo
{
    // Type of vertex :
    //  * -1 = free
    //  *  0 = fixed
    //  * >0 = boundary
    int type;

    // Neighbors list start location
    int neigBase;

    // Neighbors count
    int neigCount;

    int pad0;

    inline GpuTopo() : type(0), neigBase(0), neigCount(0) {}
    inline GpuTopo(int type, int neigBase, int neigCount) :
        type(type), neigBase(neigBase), neigCount(neigCount) {}
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

protected:
    virtual void uploadElement(
            cellar::GlProgram& program,
            const std::string& prefix,
            int edgeCount, const MeshEdge edges[],
            int triCount,  const MeshTri tris[],
            int tetCount,  const MeshTet tets[]) const;

    GLuint _vertSsbo;
    GLuint _topoSsbo;
    GLuint _neigSsbo;
    GLuint _tetSsbo;
    GLuint _priSsbo;
    GLuint _hexSsbo;
};




// IMPLEMENTATION //

#endif // GPUMESH_GPUMESH
