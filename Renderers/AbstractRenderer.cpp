#include "AbstractRenderer.h"

#include <CellarWorkbench/Misc/Log.h>

using namespace cellar;


AbstractRenderer::AbstractRenderer() :
    _buffNeedUpdate(false),
    _isPhysicalCut(true),
    _cutPlane(0.0),
    _shadingFuncs("Shadings")
{

}

AbstractRenderer::~AbstractRenderer()
{

}


void AbstractRenderer::setup()
{
    resetResources();
    setupShaders();
}

void AbstractRenderer::tearDown()
{
    clearResources();
}

void AbstractRenderer::notifyMeshUpdate()
{
    _buffNeedUpdate = true;
}

void AbstractRenderer::display(const Mesh& mesh, const AbstractEvaluator& evaluator)
{
    if(_buffNeedUpdate)
    {
        updateGeometry(mesh, evaluator);
    }

    render();
}

OptionMapDetails AbstractRenderer::availableShadings() const
{
    return _shadingFuncs.details();
}

void AbstractRenderer::useShading(const std::string& shadingName)
{
    ShadingFunc shadingFunc;
    if(_shadingFuncs.select(shadingName, shadingFunc))
        shadingFunc();
}

void AbstractRenderer::useVirtualCutPlane(bool use)
{
    _isPhysicalCut = !use;
    updateCutPlane(_cutPlane);
    _buffNeedUpdate = true;
}
