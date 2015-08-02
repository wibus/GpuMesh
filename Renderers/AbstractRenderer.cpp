#include "AbstractRenderer.h"

#include <CellarWorkbench/Misc/Log.h>

#include <Scaena/StageManagement/Event/KeyboardEvent.h>

using namespace cellar;


AbstractRenderer::AbstractRenderer() :
    _buffNeedUpdate(false),
    _cutPlaneEq(0.0),
    _physicalCutPlane(0.0),
    _virtualCutPlane(0.0),
    _cutType(ECutType::None),
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

void AbstractRenderer::useCutType(const ECutType& cutType)
{
    _cutType = cutType;
    _buffNeedUpdate = true;
    updateCutPlane(_cutPlaneEq);
}

void AbstractRenderer::handleKeyPress(const scaena::KeyboardEvent& event)
{
    if(event.getAscii() == 'C')
    {
        const char* rep = "Invalide cut type";

        switch(_cutType)
        {
        case ECutType::None :
            useCutType(ECutType::VirtualPlane);
            rep = "Virtual Plane";
            break;
        case ECutType::VirtualPlane :
            useCutType(ECutType::PhysicalPlane);
            rep = "Physical Plane";
            break;
        case ECutType::PhysicalPlane :
            useCutType(ECutType::InvertedElements);
            rep = "Inverted Elements";
            break;
        case ECutType::InvertedElements :
            useCutType(ECutType::None);
            rep = "None";
            break;
        }

        getLog().postMessage(new Message('I', false,
            std::string("Physical cut : ") + rep, "AbstractRenderer"));
    }
}
