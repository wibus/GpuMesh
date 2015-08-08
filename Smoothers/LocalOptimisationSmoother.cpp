#include "LocalOptimisationSmoother.h"

#include "OptimizationHelper.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


LocalOptimisationSmoother::LocalOptimisationSmoother() :
    AbstractSmoother({":/shaders/compute/Smoothing/LocalOptimisation.glsl"})
{

}

LocalOptimisationSmoother::~LocalOptimisationSmoother()
{

}

void LocalOptimisationSmoother::smoothVertices(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{
    for(int v = first; v < last; ++v)
    {

    }
}
