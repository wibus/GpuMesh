#ifndef GPUMESH_LOCALOPTIMISATIONSMOOTHER
#define GPUMESH_LOCALOPTIMISATIONSMOOTHER


#include "AbstractVertexWiseSmoother.h"


class LocalOptimisationSmoother : public AbstractVertexWiseSmoother
{
public:
    LocalOptimisationSmoother();
    virtual ~LocalOptimisationSmoother();

protected:
    virtual void smoothVertices(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) override;
};

#endif // GPUMESH_LOCALOPTIMISATIONSMOOTHER