#ifndef GPUMESH_GPULAPLACIANSMOOTHER
#define GPUMESH_GPULAPLACIANSMOOTHER

#include <CellarWorkbench/GL/GlProgram.h>

#include "AbstractSmoother.h"


class GpuLaplacianSmoother : public AbstractSmoother
{
public:
    GpuLaplacianSmoother(Mesh& mesh, double moveFactor, double gainThreshold);
    virtual ~GpuLaplacianSmoother();

    virtual void smoothMesh() override;

protected:
    void initializeProgram();
    void updateTopology();

    bool _initialized;
    bool _topologyChanged;
    cellar::GlProgram _smoothingProgram;

    GLuint _vertSsbo;
    GLuint _topoSsbo;
    GLuint _neigSsbo;
};

#endif // GPUMESH_GPULAPLACIANSMOOTHER
