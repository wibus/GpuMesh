#ifndef GPUMESH_GPULAPLACIANSMOOTHER
#define GPUMESH_GPULAPLACIANSMOOTHER

#include <CellarWorkbench/GL/GlProgram.h>

#include "AbstractSmoother.h"


class GpuLaplacianSmoother : public AbstractSmoother
{
public:
    GpuLaplacianSmoother(
            double moveFactor,
            double gainThreshold);
    virtual ~GpuLaplacianSmoother();

    virtual void smoothMesh(Mesh& mesh, AbstractEvaluator& evaluator) override;

protected:
    virtual void initializeProgram(Mesh& mesh);

    bool _initialized;
    cellar::GlProgram _smoothingProgram;

    std::vector<glm::vec4> _vertTmpBuff;
    size_t _vertTmpBuffSize;
};

#endif // GPUMESH_GPULAPLACIANSMOOTHER
