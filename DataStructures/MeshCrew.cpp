#include "MeshCrew.h"

#include <CellarWorkbench/GL/GlProgram.h>

#include "Mesh.h"
#include "Boundaries/AbstractBoundary.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/MetricFreeMeasurer.h"
#include "Measurers/MetricWiseMeasurer.h"
#include "Samplers/AbstractSampler.h"
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
    _sampler->installPlugin(mesh, program);
    _measurer->installPlugin(mesh, program);
    _evaluator->installPlugin(mesh, program);
}

void MeshCrew::setPluginGlslUniforms(const Mesh& mesh, cellar::GlProgram& program) const
{
    _sampler->setPluginGlslUniforms(mesh, program);
    _measurer->setPluginGlslUniforms(mesh, program);
    _evaluator->setPluginGlslUniforms(mesh, program);
}

void MeshCrew::setPluginCudaUniforms(const Mesh& mesh) const
{
    _sampler->setPluginCudaUniforms(mesh);
    _measurer->setPluginCudaUniforms(mesh);
    _evaluator->setPluginCudaUniforms(mesh);
}


void MeshCrew::reinitCrew(const Mesh& mesh)
{
    if(_isInitialized)
    {
        if(_sampler.get()   != nullptr &&
           _measurer.get()  != nullptr &&
           _evaluator.get() != nullptr)
        {
            _sampler->initialize();
            _measurer->initialize();
            _evaluator->initialize(mesh, *this);
        }
    }
}
