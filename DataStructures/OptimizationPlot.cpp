#include "OptimizationPlot.h"


OptimizationPlot::OptimizationPlot()
{

}

OptimizationPlot::~OptimizationPlot()
{

}

void OptimizationPlot::setMeshModelName(const std::string& name)
{
    _meshModelName = name;
}

void OptimizationPlot::setSmoothingMethodName(const std::string& name)
{
    _smoothingMethodName = name;
}

void OptimizationPlot::addMeshProperty(const std::string& name, const std::string& value)
{
    _meshProperties.push_back(make_pair(name, value));
}

void OptimizationPlot::addSmoothingProperty(const std::string& name, const std::string& value)
{
    _smoothingProperties.push_back(make_pair(name, value));
}

void OptimizationPlot::setInitialHistogram(const QualityHistogram& histogram)
{
    _initialHistogram = histogram;
}

void OptimizationPlot::addImplementation(const OptimizationImpl& impl)
{
    _implementations.push_back(impl);
}

const std::string& OptimizationPlot::meshModelName() const
{
    return _meshModelName;
}

const std::string& OptimizationPlot::smoothingMethodName() const
{
    return _smoothingMethodName;
}

const Properties& OptimizationPlot::meshProperties() const
{
    return _meshProperties;
}

const Properties& OptimizationPlot::smoothingProperties() const
{
    return _smoothingProperties;
}

const QualityHistogram& OptimizationPlot::initialHistogram() const
{
    return _initialHistogram;
}

const std::vector<OptimizationImpl>& OptimizationPlot::implementations() const
{
    return _implementations;
}
