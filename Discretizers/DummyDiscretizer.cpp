#include "DummyDiscretizer.h"

#include "DataStructures/Mesh.h"


DummyDiscretizer::DummyDiscretizer() :
    AbstractDiscretizer("Dummy", ":/glsl/compute/Discretizing/Dummy.glsl"),
    _debugMesh(new Mesh())
{
    _debugMesh->modelName = "Dummy discretization mesh";
}

DummyDiscretizer::~DummyDiscretizer()
{

}

bool DummyDiscretizer::isMetricWise() const
{
    return false;
}

void DummyDiscretizer::discretize(
        const Mesh& mesh,
        int density)
{

}

Metric DummyDiscretizer::metricAt(
        const glm::dvec3& position) const
{
    // Constant isotropic metric
    return Metric(1.0);
}

void DummyDiscretizer::releaseDebugMesh()
{
    // Mesh is not big
    // Never release it
}

const Mesh& DummyDiscretizer::debugMesh()
{
    return *_debugMesh;
}
