#include "MetricWiseMeasurer.h"

#include "DataStructures/Mesh.h"

#include "Discretizers/AbstractDiscretizer.h"
#include "Evaluators/AbstractEvaluator.h"


MetricWiseMeasurer::MetricWiseMeasurer() :
    AbstractMeasurer("Metric Wise", ":/shaders/compute/Measuring/MetricWise.glsl")
{

}

MetricWiseMeasurer::~MetricWiseMeasurer()
{

}

double MetricWiseMeasurer::computeLocalElementSize(
        const Mesh& mesh,
        size_t vId) const
{
    return 1.0;
}

glm::dvec3 MetricWiseMeasurer::computeVertexEquilibrium(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        size_t vId) const
{
    return mesh.verts[vId].p;
}

double MetricWiseMeasurer::computePatchQuality(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator,
            size_t vId) const
{
    return 1.0;
}

glm::dvec3 MetricWiseMeasurer::computeSpringForce(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3& pi,
        const glm::dvec3& pj) const
{
    if(pi == pj)
        return glm::dvec3();

    double d = discretizer.distance(pi, pj);
    glm::dvec3 u = (pi - pj) / d;

    double d2 = d * d;
    double d4 = d2 * d2;

    //double f = (1 - d4) * glm::exp(-d4);
    double f = (1-d2)*glm::exp(-d2/4.0)/2.0;

    return f * u;
}
