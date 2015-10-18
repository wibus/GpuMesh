#include "MeshCrew.h"

#include <CellarWorkbench/GL/GlProgram.h>

#include "Mesh.h"
#include "Discretizers/AbstractDiscretizer.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/MetricFreeMeasurer.h"
#include "Measurers/MetricWiseMeasurer.h"


const std::string METRIC_FREE = "Metric Free";
const std::string METRIC_WISE = "Metric Wise";


MeshCrew::MeshCrew() :
    _availableMeasurers("Available Measurers"),
    _isInitialized(false)
{
    _availableMeasurers.setDefault(METRIC_WISE);
    _availableMeasurers.setContent({
        {METRIC_FREE, std::shared_ptr<AbstractMeasurer>(new MetricFreeMeasurer())},
        {METRIC_WISE, std::shared_ptr<AbstractMeasurer>(new MetricWiseMeasurer())}
    });
}

void MeshCrew::initialize(const Mesh& mesh)
{
    _isInitialized = true;
    reinitCrew(mesh);
}

void MeshCrew::terminate()
{
    _isInitialized = false;
}

bool MeshCrew::initialized() const
{
    return _isInitialized;
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

void MeshCrew::setDiscretizer(const Mesh& mesh, const std::shared_ptr<AbstractDiscretizer>& discretizer)
{
    _discretizer = discretizer;

    if(discretizer->isMetricWise())
        _availableMeasurers.select(METRIC_WISE, _measurer);
    else
        _availableMeasurers.select(METRIC_FREE, _measurer);

    reinitCrew(mesh);
}

void MeshCrew::setEvaluator(const Mesh& mesh, const std::shared_ptr<AbstractEvaluator>& evaluator)
{
    _evaluator = evaluator;

    reinitCrew(mesh);
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

void MeshCrew::reinitCrew(const Mesh& mesh)
{
    if(_isInitialized)
    {
        if(_discretizer.get() != nullptr &&
           _measurer.get()    != nullptr &&
           _evaluator.get()   != nullptr)
        {
            _evaluator->initialize(mesh, discretizer(), measurer());
        }
    }
}
