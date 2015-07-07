#ifndef GPUMESH_GPUMESH
#define GPUMESH_GPUMESH

#include <GL3/gl3w.h>

#include "Mesh.h"


struct GpuTopo
{
    // Type of vertex :
    //  * -1 = free
    //  *  0 = fixed
    //  * >0 = boundary
    int type;

    // Neighbors list start location
    int base;

    // Neighbors count
    int count;

    int pad;

    GpuTopo() : type(0), base(0), count(0) {}
    GpuTopo(int type, int base, int count) :
        type(type), base(base), count(count) {}
};

struct GpuQual
{
    unsigned int mean;
    unsigned int var;
    unsigned int min;
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

    virtual unsigned int glBuffer(const EMeshBuffer& buffer) const override;


protected:
    GLuint _vertSsbo;
    GLuint _topoSsbo;
    GLuint _neigSsbo;
    GLuint _qualSsbo;
    GLuint _tetSsbo;
    GLuint _priSsbo;
    GLuint _hexSsbo;
};




// IMPLEMENTATION //

#endif // GPUMESH_GPUMESH
