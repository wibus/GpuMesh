#ifndef GPUMESH_QUALITYLAPLACESMOOTHER
#define GPUMESH_QUALITYLAPLACESMOOTHER


#include "AbstractSmoother.h"


class QualityLaplaceSmoother : public AbstractSmoother
{
public:
    QualityLaplaceSmoother();
    virtual ~QualityLaplaceSmoother();

    virtual void smoothMeshSerial(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;
};

#endif // GPUMESH_QUALITYLAPLACESMOOTHER
