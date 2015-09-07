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
            const std::vector<uint>& vIds) override;
};

#endif // GPUMESH_SPRINGLAPLACESMOOTHER
