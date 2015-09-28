#include "AbstractRenderer.h"

#include <CellarWorkbench/Misc/Log.h>
#include <CellarWorkbench/Image/Image.h>
#include <CellarWorkbench/Image/ImageBank.h>
#include <CellarWorkbench/GL/GlToolkit.h>

#include <Scaena/StageManagement/Event/KeyboardEvent.h>

using namespace cellar;


AbstractRenderer::AbstractRenderer() :
    _buffNeedUpdate(false),
    _cutType(ECutType::None),
    _cutPlaneEq(0.0),
    _physicalCutPlane(0.0),
    _virtualCutPlane(0.0),
    _tetVisibility(true),
    _priVisibility(true),
    _hexVisibility(true),
    _qualityCullingMin(-INFINITY),
    _qualityCullingMax(INFINITY),
    _shadingFuncs("Shadings"),
    _fullscreenVao(0),
    _fullscreenVbo(0),
    _filterTex(0),
    _filterWidth(1),
    _filterHeight(1),
    _filterScale(1, 1)
{

}

AbstractRenderer::~AbstractRenderer()
{

}

void AbstractRenderer::notify(cellar::CameraMsg& msg)
{
    if(msg.change == CameraMsg::EChange::VIEWPORT)
    {
        const glm::ivec2& viewport = msg.camera.viewport();

        // Background scale
        glm::vec2 viewportf(viewport);
        glm::vec2 backSize(_filterWidth,
                           _filterHeight);
        _filterScale = viewportf / backSize;
        if(_filterScale.x > 1.0)
            _filterScale /= _filterScale.x;
        if(_filterScale.y > 1.0)
            _filterScale /= _filterScale.y;

        _gradientShader.pushProgram();
        _gradientShader.setVec2f("TexScale", _filterScale);
        _gradientShader.popProgram();
    }

    notifyCameraUpdate(msg);
}

void AbstractRenderer::setup()
{
    resetResources();
    setupShaders();


    // Backdrop
    GLfloat backTriangle[] = {
        -1.0f, -1.0f,
         3.0f, -1.0f,
        -1.0f,  3.0f
    };

    glGenVertexArrays(1, &_fullscreenVao);
    glBindVertexArray(_fullscreenVao);

    glGenBuffers(1, &_fullscreenVbo);
    glBindBuffer(GL_ARRAY_BUFFER, _fullscreenVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(backTriangle), backTriangle, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    _gradientShader.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/Filter.vert");
    _gradientShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/Gradient.frag");
    _gradientShader.link();
    _gradientShader.pushProgram();
    _gradientShader.setInt("Filter", 1);
    _gradientShader.setVec2f("TexScale", _filterScale);
    _gradientShader.popProgram();

    Image& filterTex =  getImageBank().getImage("resources/textures/Filter.png");
    _filterTex = GlToolkit::genTextureId(filterTex);
    _filterWidth = filterTex.width();
    _filterHeight = filterTex.height();
}

void AbstractRenderer::tearDown()
{
    clearResources();

    glDeleteVertexArrays(1, &_fullscreenVao);
    _fullscreenVao = 0;

    glDeleteBuffers(1, &_fullscreenVbo);
    _fullscreenVbo = 0;

    GlToolkit::deleteTextureId(_filterTex);
    _filterTex = 0;
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

void AbstractRenderer::setElementVisibility(bool tet, bool pri, bool hex)
{
    _tetVisibility = tet;
    _priVisibility = pri;
    _hexVisibility = hex;
    notifyMeshUpdate();
}

void AbstractRenderer::setQualityCullingBounds(double min, double max)
{
    _qualityCullingMin = min;
    _qualityCullingMax = max;
    notifyMeshUpdate();
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

void AbstractRenderer::drawBackdrop()
{
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _filterTex);
    glActiveTexture(GL_TEXTURE0);

    _gradientShader.pushProgram();
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);

    fullScreenDraw();

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    _gradientShader.popProgram();
}

void AbstractRenderer::fullScreenDraw()
{
    glBindVertexArray(_fullscreenVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
