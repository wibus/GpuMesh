#ifndef GPUMESH_SPRINGLAPLACESMOOTHER
#define GPUMESH_SPRINGLAPLACESMOOTHER


#include "AbstractVertexWiseSmoother.h"


class SpringLaplaceSmoother : public AbstractVertexWiseSmoother
{
public:
    SpringLaplaceSmoother();
    virtual ~SpringLaplaceSmoother();

protected:
    virtual void smoothVertices(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) override;
};

#endif // GPUMESH_SPRINGLAPLACESMOOTHER
