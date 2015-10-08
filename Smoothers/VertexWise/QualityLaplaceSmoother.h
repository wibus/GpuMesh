#ifndef GPUMESH_QUALITYLAPLACESMOOTHER
#define GPUMESH_QUALITYLAPLACESMOOTHER


#include "AbstractVertexWiseSmoother.h"


class QualityLaplaceSmoother : public AbstractVertexWiseSmoother
{
public:
    QualityLaplaceSmoother();
    virtual ~QualityLaplaceSmoother();

protected:
    virtual void printSmoothingParameters(
            const Mesh& mesh,
            OptimizationPlot& plot) const override;

    virtual void smoothVertices(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            const AbstractDiscretizer& discretizer,
            const std::vector<uint>& vIds) override;
};

#endif // GPUMESH_QUALITYLAPLACESMOOTHER
