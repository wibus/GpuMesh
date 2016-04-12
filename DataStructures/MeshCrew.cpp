#include "MeshCrew.h"

#include <CellarWorkbench/GL/GlProgram.h>

#include "Mesh.h"
#include "Samplers/AbstractSampler.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/MetricFreeMeasurer.h"
#include "Measurers/MetricWiseMeasurer.h"
#include "Topologists/BatrTopologist.h"


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

    _topologist.reset(new BatrTopologist());
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

AbstractMeasurer& MeshCrew::measurer()
{
    return *_measurer;
}

AbstractEvaluator& MeshCrew::evaluator()
{
    return *_evaluator;
}

AbstractTopologist& MeshCrew::topologist()
{
    return *_topologist;
}

const AbstractSampler& MeshCrew::sampler() const
{
    return *_sampler;
}

const AbstractMeasurer& MeshCrew::measurer() const
{
    return *_measurer;
}

const AbstractEvaluator& MeshCrew::evaluator() const
{
    return *_evaluator;
}

const AbstractTopologist& MeshCrew::topologist() const
{
    return *_topologist;
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

bool MeshCrew::needTopologicalModifications(int pass) const
{
    return (pass > 1) &&
           (_topologist->isEnabled()) &&
           ((pass-1) % _topologist->frequency() == 0);
}

void MeshCrew::enableTopologyModifications(bool enable)
{
    _topologist->setEnabled(enable);
}

void MeshCrew::setTopologyModificationsFrequency(int frequency)
{
    _topologist->setFrequency(frequency);
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
