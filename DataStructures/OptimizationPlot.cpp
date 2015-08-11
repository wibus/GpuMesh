#include "OptimizationPlot.h"


OptimizationPlot::OptimizationPlot(const std::string& title) :
    _title(title)
{

}

OptimizationPlot::~OptimizationPlot()
{

}

void OptimizationPlot::clear()
{
    _curves.clear();
}

void OptimizationPlot::addCurve(
        const std::string label,
        const OptimizationPassVect& passes)
{
    _curves.insert(make_pair(label, passes));
}

const std::string& OptimizationPlot::title() const
{
    return _title;
}

const OptimizationCurvesMap& OptimizationPlot::curves() const
{
    return _curves;
}
