#ifndef GPUMESH_GETMESMOOTHER
#define GPUMESH_GETMESMOOTHER


#include "AbstractElementWiseSmoother.h"


class GetmeSmoother : public AbstractElementWiseSmoother
{
public:
    GetmeSmoother();
    virtual ~GetmeSmoother();

protected:
    virtual void smoothTets(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) override;

    virtual void smoothPris(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) override;

    virtual void smoothHexs(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) override;
};

#endif // GPUMESH_GETMESMOOTHER
