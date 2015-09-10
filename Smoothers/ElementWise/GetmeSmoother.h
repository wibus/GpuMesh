#ifndef GPUMESH_GETMESMOOTHER
#define GPUMESH_GETMESMOOTHER


#include "AbstractElementWiseSmoother.h"



class GetmeSmoother : public AbstractElementWiseSmoother
{
public:
    GetmeSmoother();
    virtual ~GetmeSmoother();

protected:
    virtual void smoothTets(Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last) override;

    virtual void smoothPris(Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last) override;

    virtual void smoothHexs(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last) override;

protected:
    virtual void printImplParameters(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator,
            OptimizationImpl& implementation) const override;

private:
    double _lambda;
};

#endif // GPUMESH_GETMESMOOTHER
