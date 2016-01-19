#include "AnalyticDiscretizer.h"

#include "DataStructures/Mesh.h"


AnalyticDiscretizer::AnalyticDiscretizer() :
    AbstractDiscretizer("Analytic", ":/glsl/compute/Discretizing/Analytic.glsl"),
    _debugMesh(new Mesh())
{
    _debugMesh->modelName = "Analytic discretization mesh";
}

AnalyticDiscretizer::~AnalyticDiscretizer()
{

}

bool AnalyticDiscretizer::isMetricWise() const
{
    return true;
}

void AnalyticDiscretizer::discretize(
        const Mesh& mesh,
        int density)
{

}

Metric AnalyticDiscretizer::metricAt(
        const glm::dvec3& position) const
{
    return vertMetric(position);
}

void AnalyticDiscretizer::releaseDebugMesh()
{
}

const Mesh& AnalyticDiscretizer::debugMesh()
{
    return *_debugMesh;
}
