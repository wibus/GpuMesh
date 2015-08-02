#ifndef GPUMESH_SPRINGLAPLACESMOOTHER
#define GPUMESH_SPRINGLAPLACESMOOTHER


#include "AbstractSmoother.h"


class SpringLaplaceSmoother : public AbstractSmoother
{
public:
    SpringLaplaceSmoother();
    virtual ~SpringLaplaceSmoother();

    virtual void smoothMeshCpp(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;
};

#endif // GPUMESH_SPRINGLAPLACESMOOTHER