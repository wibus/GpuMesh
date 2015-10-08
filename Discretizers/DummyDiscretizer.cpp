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

Metric DummyDiscretizer::metric(
        const glm::dvec3& position) const
{
    // Constant isotropic metric
    return Metric(1.0);
}

double DummyDiscretizer::distance(
        const glm::dvec3& a,
        const glm::dvec3& b) const
{
    return 1.0;//glm::distance(a, b);
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
