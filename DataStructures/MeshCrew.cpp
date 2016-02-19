#include "MeshCrew.h"

#include <CellarWorkbench/GL/GlProgram.h>

#include "Mesh.h"
#include "Samplers/AbstractSampler.h"
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

AbstractSampler& MeshCrew::sampler()
{
    return *_sampler;
}

AbstractEvaluator& MeshCrew::evaluator()
{
    return *_evaluator;
}

AbstractMeasurer& MeshCrew::measurer()
{
    return *_measurer;
}

const AbstractSampler& MeshCrew::sampler() const
{
    return *_sampler;
}

const AbstractEvaluator& MeshCrew::evaluator() const
{
    return *_evaluator;
}

const AbstractMeasurer& MeshCrew::measurer() const
{
    return *_measurer;
}

void MeshCrew::setSampler(const Mesh& mesh, const std::shared_ptr<AbstractSampler>& sampler)
{
    _sampler = sampler;

    if(sampler->isMetricWise())
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

void MeshCrew::installPlugins(const Mesh& mesh, cellar::GlProgram& program) const
{
    // Mesh's plugin
    program.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        mesh.modelBoundsShaderName().c_str()});

    mesh.modelBoundsCudaFct()();

    // Crew members' plugin
    _sampler->installPlugin(mesh, program);
    _evaluator->installPlugin(mesh, program);
    _measurer->installPlugin(mesh, program);
}

void MeshCrew::setPluginUniforms(const Mesh& mesh, cellar::GlProgram& program) const
{
    // Mesh's uniforms
    mesh.uploadGeometry(program);

    // Crew members' uniforms
    _sampler->setPluginUniforms(mesh, program);
    _evaluator->setPluginUniforms(mesh, program);
    _measurer->setPluginUniforms(mesh, program);
}

void MeshCrew::setupPluginExecution(const Mesh& mesh, const cellar::GlProgram& program) const
{
    // Mesh's uniforms
    mesh.bindShaderStorageBuffers();

    // Crew members' buffers
    _sampler->setupPluginExecution(mesh, program);
    _evaluator->setupPluginExecution(mesh, program);
    _measurer->setupPluginExecution(mesh, program);
}

void MeshCrew::reinitCrew(const Mesh& mesh)
{
    if(_isInitialized)
    {
        if(_sampler.get() != nullptr &&
           _measurer.get()    != nullptr &&
           _evaluator.get()   != nullptr)
        {
            _sampler->initialize();
            _measurer->initialize();
            _evaluator->initialize(mesh, *this);
        }
    }
}
