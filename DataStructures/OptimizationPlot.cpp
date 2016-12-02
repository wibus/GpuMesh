#include "OptimizationPlot.h"


void OptimizationImpl::addSmoothingProperty(const std::string& name, const std::string& value)
{
    smoothingProperties.push_back(make_pair(name, value));
}


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

void OptimizationPlot::setNodeGroups(const NodeGroups& groups)
{
    _nodeGroups = groups;
}

void OptimizationPlot::addMeshProperty(const std::string& name, const std::string& value)
{
    _meshProperties.push_back(make_pair(name, value));
}

void OptimizationPlot::setInitialHistogram(const QualityHistogram& histogram)
{
    _initialHistogram = histogram;
}

void OptimizationPlot::addImplementation(const OptimizationImpl& impl)
{
    _implementations.push_back(impl);
}

const NodeGroups& OptimizationPlot::nodeGroups() const
{
    return _nodeGroups;
}

const Properties& OptimizationPlot::meshProperties() const
{
    return _meshProperties;
}

const std::string& OptimizationPlot::meshModelName() const
{
    return _meshModelName;
}

const QualityHistogram& OptimizationPlot::initialHistogram() const
{
    return _initialHistogram;
}

const std::vector<OptimizationImpl>& OptimizationPlot::implementations() const
{
    return _implementations;
}
