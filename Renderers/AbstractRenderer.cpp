#include "AbstractRenderer.h"


AbstractRenderer::AbstractRenderer() :
    _buffNeedUpdate(false)
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