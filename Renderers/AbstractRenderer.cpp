#include "AbstractRenderer.h"

#include <CellarWorkbench/Misc/Log.h>

using namespace cellar;


AbstractRenderer::AbstractRenderer() :
    _buffNeedUpdate(false),
    _isPhysicalCut(true),
    _cutPlane(0.0)
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

std::vector<std::string> AbstractRenderer::availableShadings() const
{
    std::vector<std::string> names;
    for(const auto& keyValue : _shadingFuncs)
        names.push_back(keyValue.first);
    return names;
}

void AbstractRenderer::useShading(const std::string& shadingName)
{
    auto it = _shadingFuncs.find(shadingName);
    if(it != _shadingFuncs.end())
    {
        it->second();
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Failed to find '" + shadingName + "' shading", "AbstractRenderer"));
    }
}

void AbstractRenderer::useVirtualCutPlane(bool use)
{
    _isPhysicalCut = !use;
    updateCutPlane(_cutPlane);
    _buffNeedUpdate = true;
}
