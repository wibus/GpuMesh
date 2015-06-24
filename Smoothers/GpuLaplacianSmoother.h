#ifndef GPUMESH_GPULAPLACIANSMOOTHER
#define GPUMESH_GPULAPLACIANSMOOTHER


#include "AbstractSmoother.h"


class GpuLaplacianSmoother : public AbstractSmoother
{
public:
    GpuLaplacianSmoother(Mesh& mesh, double moveFactor, double gainThreshold);
    virtual ~GpuLaplacianSmoother();

    virtual void smoothMesh() override;
};

#endif // GPUMESH_GPULAPLACIANSMOOTHER
