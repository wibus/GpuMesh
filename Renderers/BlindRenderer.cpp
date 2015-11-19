#include "BlindRenderer.h"

using namespace std;


BlindRenderer::BlindRenderer()
{
    _shadingFuncs.setDefault("No Shading");
    _shadingFuncs.setContent({
        {string("No Shading"), function<void()>(bind(&BlindRenderer::useNoShading, this))},
    });
}

BlindRenderer::~BlindRenderer()
{

}

void BlindRenderer::updateCamera(
        const glm::mat4& view,
        const glm::vec3& pos)
{

}

void BlindRenderer::updateLight(
        const glm::mat4& view,
        const glm::vec3& pos)
{

}

void BlindRenderer::updateCutPlane(
        const glm::dvec4& cutEq)
{

}

bool BlindRenderer::handleKeyPress(
        const scaena::KeyboardEvent& event)
{
	return false;
}

void BlindRenderer::handleInputs(
        const scaena::SynchronousKeyboard& keyboard,
        const scaena::SynchronousMouse& mouse)
{

}

void BlindRenderer::updateGeometry(const Mesh& mesh)
{

}

void BlindRenderer::notifyCameraUpdate(
        cellar::CameraMsg& msg)
{

}

void BlindRenderer::clearResources()
{

}

void BlindRenderer::resetResources()
{

}

void BlindRenderer::setupShaders()
{

}

void BlindRenderer::render()
{
    drawBackdrop();
}

void BlindRenderer::useNoShading()
{

}
