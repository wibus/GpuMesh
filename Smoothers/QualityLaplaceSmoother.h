#ifndef GPUMESH_CPULAPLACIANSMOOTHER
#define GPUMESH_CPULAPLACIANSMOOTHER


#include "AbstractSmoother.h"


class QualityLaplaceSmoother : public AbstractSmoother
{
public:
    QualityLaplaceSmoother();
    virtual ~QualityLaplaceSmoother();

    virtual void smoothCpuMesh(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;
};

#endif // GPUMESH_CPULAPLACIANSMOOTHER
