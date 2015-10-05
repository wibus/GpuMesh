#include "DummyDiscretizer.h"

#include "DataStructures/Mesh.h"


DummyDiscretizer::DummyDiscretizer() :
    _debugMesh(new Mesh())
{
    _debugMesh->modelName = "Dummy discretization mesh";
}

DummyDiscretizer::~DummyDiscretizer()
{

}

void DummyDiscretizer::discretize(
        const Mesh& mesh,
        const glm::ivec3& gridSize)
{

}

Metric DummyDiscretizer::metricAt(
        const glm::dvec3& position) const
{
    return Metric(1.0);
}

void DummyDiscretizer::installPlugIn(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{

}

void DummyDiscretizer::uploadPlugInUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{

}

void DummyDiscretizer::releaseDebugMesh()
{
    // Mesh is not big
    // Never release it
}

std::shared_ptr<Mesh> DummyDiscretizer::debugMesh()
{
    return _debugMesh;
}
