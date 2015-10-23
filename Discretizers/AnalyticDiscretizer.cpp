#include "AnalyticDiscretizer.h"

#include "DataStructures/Mesh.h"


AnalyticDiscretizer::AnalyticDiscretizer() :
    AbstractDiscretizer("Analytic", ""),
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

void AnalyticDiscretizer::installPlugIn(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    AbstractDiscretizer::installPlugIn(mesh, program);
}

void AnalyticDiscretizer::uploadUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    AbstractDiscretizer::uploadUniforms(mesh, program);
}

void AnalyticDiscretizer::discretize(
        const Mesh& mesh,
        int density)
{

}

Metric AnalyticDiscretizer::metric(
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
