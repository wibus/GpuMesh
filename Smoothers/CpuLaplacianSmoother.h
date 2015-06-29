#ifndef GPUMESH_CPULAPLACIANSMOOTHER
#define GPUMESH_CPULAPLACIANSMOOTHER


#include "AbstractSmoother.h"


class CpuLaplacianSmoother : public AbstractSmoother
{
public:
    CpuLaplacianSmoother(
            double moveFactor,
            double gainThreshold);
    virtual ~CpuLaplacianSmoother();

    virtual void smoothMesh(Mesh& mesh, AbstractEvaluator& evaluator) override;
};

#endif // GPUMESH_CPULAPLACIANSMOOTHER
