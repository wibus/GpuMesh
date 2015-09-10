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

void OptimizationPlot::addImplementation(const OptimizationImpl& impl)
{
    _implementations.push_back(impl);
}

void OptimizationPlot::addMeshProperty(const std::string& name, const std::string& value)
{
    _meshProperties[name] = value;
}

const std::string& OptimizationPlot::meshModelName() const
{
    return _meshModelName;
}

const std::string& OptimizationPlot::smoothingMethodName() const
{
    return _smoothingMethodName;
}

const std::vector<OptimizationImpl>& OptimizationPlot::implementations() const
{
    return _implementations;
}

const std::map<std::string, std::string>& OptimizationPlot::meshProperties() const
{
    return _meshProperties;
}
