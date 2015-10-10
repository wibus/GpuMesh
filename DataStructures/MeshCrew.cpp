#include "MeshCrew.h"

#include <CellarWorkbench/GL/GlProgram.h>

#include "Mesh.h"
#include "Discretizers/AbstractDiscretizer.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"


MeshCrew::MeshCrew()
{

}

AbstractDiscretizer& MeshCrew::discretizer()
{
    return *_discretizer;
}

AbstractEvaluator& MeshCrew::evaluator()
{
    return *_evaluator;
}

AbstractMeasurer& MeshCrew::measurer()
{
    return *_measurer;
}

const AbstractDiscretizer& MeshCrew::discretizer() const
{
    return *_discretizer;
}

const AbstractEvaluator& MeshCrew::evaluator() const
{
    return *_evaluator;
}

const AbstractMeasurer& MeshCrew::measurer() const
{
    return *_measurer;
}

void MeshCrew::setDiscretizer(const std::shared_ptr<AbstractDiscretizer>& discretizer)
{
    _discretizer = discretizer;
}

void MeshCrew::setEvaluator(const std::shared_ptr<AbstractEvaluator>& evaluator)
{
    _evaluator = evaluator;
}

void MeshCrew::setMeasurer(const std::shared_ptr<AbstractMeasurer>& measurer)
{
    _measurer = measurer;
}

void MeshCrew::installPlugIns(const Mesh& mesh, cellar::GlProgram& program) const
{
    // Mesh's plugin
    program.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        mesh.modelBoundsShaderName().c_str()});

    // Crew members' plugin
    _discretizer->installPlugIn(mesh, program);
    _evaluator->installPlugIn(mesh, program);
    _measurer->installPlugIn(mesh, program);
}

void MeshCrew::uploadUniforms(const Mesh& mesh, cellar::GlProgram& program) const
{
    // Mesh's uniforms
    mesh.uploadGeometry(program);

    // Crew members' uniforms
    _discretizer->uploadUniforms(mesh, program);
    _evaluator->uploadUniforms(mesh, program);
    _measurer->uploadUniforms(mesh, program);
}
