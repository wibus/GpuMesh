#ifndef GPUMESH_CPULAPLACIANSMOOTHER
#define GPUMESH_CPULAPLACIANSMOOTHER


#include "AbstractSmoother.h"


class CpuLaplacianSmoother : public AbstractSmoother
{
public:
    CpuLaplacianSmoother(Mesh& mesh, double moveFactor, double gainThreshold);
    virtual ~CpuLaplacianSmoother();

    virtual void smoothMesh() override;
};

#endif // GPUMESH_CPULAPLACIANSMOOTHER
