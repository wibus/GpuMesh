#include "MidEndRenderer.h"

#include <iostream>

#include <GLM/gtc/matrix_transform.hpp>

#include <CellarWorkbench/Image/Image.h>
#include <CellarWorkbench/Image/ImageBank.h>
#include <CellarWorkbench/GL/GlToolkit.h>

#include <Scaena/StageManagement/Event/KeyboardEvent.h>
#include <Scaena/StageManagement/Event/SynchronousKeyboard.h>
#include <Scaena/StageManagement/Event/SynchronousMouse.h>

#include "Evaluators/GpuEvaluator.h"

using namespace std;
using namespace cellar;


MidEndRenderer::MidEndRenderer() :
    _buffElemCount(0),
    _meshVao(0),
    _vbo(0),
    _nbo(0),
    _ebo(0),
    _qbo(0),
    _shadowFbo(0),
    _shadowDpt(0),
    _shadowTex(0),
    _bloomFbo(0),
    _bloomDpt(0),
    _bloomBaseTex(0),
    _bloomBlurTex(0),
    _useBackdrop(true),
    _fullscreenVao(0),
    _fullscreenVbo(0),
    _filterTex(0),
    _lightingEnabled(false),
    _updateShadow(false),
    _shadowSize(1024, 1024),
    _isPhysicalCut(true),
    _cutPlane(0.0),
    _evaluator(new GpuEvaluator())
{

}

MidEndRenderer::~MidEndRenderer()
{

}

void MidEndRenderer::notify(cellar::CameraMsg& msg)
{
    if(msg.change == CameraMsg::EChange::VIEWPORT)
    {
        const glm::ivec2& viewport = msg.camera.viewport();

        // Camera projection
        glm::mat4 proj = glm::perspectiveFov(
                glm::pi<float>() / 6,
                (float) viewport.x,
                (float) viewport.y,
                0.1f,
                12.0f);

        _litShader.pushProgram();
        _litShader.setMat4f("ProjMat", proj);
        _litShader.popProgram();

        _unlitShader.pushProgram();
        _unlitShader.setMat4f("ProjMat", proj);
        _unlitShader.popProgram();


        // Background scale
        glm::vec2 viewportf(viewport);
        glm::vec2 backSize(_filterWidth, _filterHeight);
        glm::vec2 scale = viewportf / backSize;
        if(scale.x > 1.0)
            scale /= scale.x;
        if(scale.y > 1.0)
            scale /= scale.y;

        _gradientShader.pushProgram();
        _gradientShader.setVec2f("TexScale", scale);
        _gradientShader.popProgram();

        _screenShader.pushProgram();
        _screenShader.setVec2f("TexScale", scale);
        _screenShader.popProgram();

        _brushShader.pushProgram();
        _brushShader.setVec2f("TexScale", scale);
        _brushShader.popProgram();

        _grainShader.pushProgram();
        _grainShader.setVec2f("TexScale", scale);
        _grainShader.popProgram();

        _updateShadow = true;

        // Resize bloom buffers
        glBindTexture(GL_TEXTURE_2D, _bloomBaseTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, viewport.x, viewport.y,
                     0, GL_RGB, GL_UNSIGNED_INT, NULL);

        glBindTexture(GL_TEXTURE_2D, _bloomBlurTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, viewport.x, viewport.y,
                     0, GL_RGB, GL_UNSIGNED_INT, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, _bloomDpt);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32,
                              viewport.x, viewport.y);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
}

void MidEndRenderer::updateCamera(const glm::mat4& view,
                                  const glm::vec3& pos)
{
    _litShader.pushProgram();
    _litShader.setMat4f("ViewMat", view);
    _litShader.setVec3f("CameraPosition", pos);
    _litShader.popProgram();

    _unlitShader.pushProgram();
    _unlitShader.setMat4f("ViewMat", view);
    _unlitShader.setVec3f("CameraPosition", pos);
    _unlitShader.popProgram();
}

void MidEndRenderer::updateLight(const glm::mat4& view,
                                 const glm::vec3& pos)
{
    glm::mat4 pvMat = _shadowProj * view;

    glm::mat4 pvShadow =
            glm::scale(glm::mat4(),     glm::vec3(0.5, 0.5, 0.5)) *
            glm::translate(glm::mat4(), glm::vec3(1.0, 1.0, 1.0)) *
            view;


    _litShader.pushProgram();
    _litShader.setVec3f("LightDirection", pos);
    _litShader.setMat4f("PVshadow", pvShadow);
    _litShader.popProgram();

    _unlitShader.pushProgram();
    _unlitShader.setVec3f("LightDirection", pos);
    _unlitShader.popProgram();

    _shadowShader.pushProgram();
    _shadowShader.setMat4f("PVmat", pvMat);
    _shadowShader.popProgram();

    _updateShadow = true;
}

void MidEndRenderer::updateCutPlane(const glm::dvec4& cutEq)
{
    _cutPlane = cutEq;
    _updateShadow = true;
    glm::vec4 virtualCut(cutEq);

    if(_isPhysicalCut)
    {
        _buffNeedUpdate = true;
        virtualCut = glm::vec4();
    }

    _litShader.pushProgram();
    _litShader.setVec4f("CutPlaneEq", virtualCut);
    _litShader.popProgram();

    _unlitShader.pushProgram();
    _unlitShader.setVec4f("CutPlaneEq",virtualCut);
    _unlitShader.popProgram();

    _shadowShader.pushProgram();
    _shadowShader.setVec4f("CutPlaneEq", virtualCut);
    _shadowShader.popProgram();
}

void MidEndRenderer::handleKeyPress(const scaena::KeyboardEvent& event)
{
    if(event.getAscii() == 'X')
    {
        _lightingEnabled = !_lightingEnabled;
        _updateShadow = true;

        const char* rep = (_lightingEnabled ? "true" : "false");
        cout << "Lighting enabled : " << rep << endl;
    }
    else if(event.getAscii() == 'C')
    {
        _isPhysicalCut = !_isPhysicalCut;
        updateCutPlane(_cutPlane);
        _buffNeedUpdate = true;

        const char* rep = (_isPhysicalCut ? "true" : "false") ;
        cout << "Physical cut : " << rep << endl;
    }
}

void MidEndRenderer::handleInputs(const scaena::SynchronousKeyboard& keyboard,
                                  const scaena::SynchronousMouse& mouse)
{

}

void MidEndRenderer::updateGeometry(const Mesh& mesh)
{
    // Clear old vertex attributes
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, _nbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, _ebo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, _qbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);


    // Fetch new vertex attributes
    vector<glm::vec3> vertices;
    vector<signed char> normals;
    vector<unsigned char> edges;
    vector<unsigned char> qualities;
    glm::dvec4 cutPlane = _isPhysicalCut ?
        _cutPlane : glm::dvec4(0, 0, 0, 0);

    mesh.compileFacesAttributes(
                *_evaluator,
                cutPlane,
                vertices,
                normals,
                edges,
                qualities);


    _buffElemCount = vertices.size();
    _buffNeedUpdate = false;


    // Send new vertex attribute
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    GLuint verticesSize = vertices.size() * sizeof(decltype(vertices.front()));
    glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices.data(), GL_STATIC_DRAW);
    vertices.clear();
    vertices.shrink_to_fit();

    glBindBuffer(GL_ARRAY_BUFFER, _nbo);
    GLuint normalsSize = normals.size() * sizeof(decltype(normals.front()));
    glBufferData(GL_ARRAY_BUFFER, normalsSize, normals.data(), GL_STATIC_DRAW);
    normals.clear();
    normals.shrink_to_fit();

    glBindBuffer(GL_ARRAY_BUFFER, _ebo);
    GLuint edgesSize = edges.size() * sizeof(decltype(edges.front()));
    glBufferData(GL_ARRAY_BUFFER, edgesSize, edges.data(), GL_STATIC_DRAW);
    edges.clear();
    edges.shrink_to_fit();

    glBindBuffer(GL_ARRAY_BUFFER, _qbo);
    GLuint qualitiesSize = qualities.size() * sizeof(decltype(qualities.front()));
    glBufferData(GL_ARRAY_BUFFER, qualitiesSize, qualities.data(), GL_STATIC_DRAW);
    qualities.clear();
    qualities.shrink_to_fit();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void MidEndRenderer::clearResources()
{
    // Delete old buffers
    glDeleteVertexArrays(1, &_meshVao);
    _meshVao = 0;

    glDeleteBuffers(1, &_vbo);
    _vbo = 0;

    glDeleteBuffers(1, &_nbo);
    _nbo = 0;

    glDeleteBuffers(1, &_ebo);
    _ebo = 0;

    glDeleteBuffers(1, &_qbo);
    _qbo = 0;

    GlToolkit::deleteTextureId(_filterTex);
    _filterTex = 0;

    glDeleteBuffers(1, &_fullscreenVao);
    _fullscreenVao = 0;

    glDeleteBuffers(1, &_fullscreenVbo);
    _fullscreenVbo = 0;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteRenderbuffers(1, &_shadowDpt);
    _shadowDpt = 0;

    glDeleteFramebuffers(1, &_shadowFbo);
    _shadowFbo = 0;

    glDeleteTextures(1, &_shadowTex);
    _shadowTex = 0;

    glDeleteRenderbuffers(1, &_bloomDpt);
    _bloomDpt = 0;

    glDeleteFramebuffers(1, &_bloomFbo);
    _bloomFbo = 0;

    glDeleteTextures(1, &_bloomBaseTex);
    _bloomBaseTex = 0;

    glDeleteTextures(1, &_bloomBlurTex);
    _bloomBlurTex = 0;


    _buffElemCount = 0;
}

void MidEndRenderer::resetResources()
{
    // Generate new buffers
    glGenVertexArrays(1, &_meshVao);
    glBindVertexArray(_meshVao);

    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &_nbo);
    glBindBuffer(GL_ARRAY_BUFFER, _nbo);
    glVertexAttribPointer(1, 3, GL_BYTE, GL_TRUE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &_ebo);
    glBindBuffer(GL_ARRAY_BUFFER, _ebo);
    glVertexAttribPointer(2, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(2);

    glGenBuffers(1, &_qbo);
    glBindBuffer(GL_ARRAY_BUFFER, _qbo);
    glVertexAttribPointer(3, 1, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);


    // Shadow casting
    glGenTextures(1, &_shadowTex);
    glBindTexture(GL_TEXTURE_2D, _shadowTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, _shadowSize.x, _shadowSize.y,
                 0, GL_RG, GL_UNSIGNED_INT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenRenderbuffers(1, &_shadowDpt);
    glBindRenderbuffer(GL_RENDERBUFFER, _shadowDpt);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, _shadowSize.x, _shadowSize.y);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glGenFramebuffers(1, &_shadowFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _shadowFbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _shadowDpt);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _shadowTex, 0);

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    // Bloom
    glGenRenderbuffers(1, &_bloomDpt);
    glBindRenderbuffer(GL_RENDERBUFFER, _bloomDpt);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, 1, 1);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);


    glGenTextures(1, &_bloomBaseTex);
    glBindTexture(GL_TEXTURE_2D, _bloomBaseTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 1, 1,
                 0, GL_RGB, GL_UNSIGNED_INT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &_bloomBlurTex);
    glBindTexture(GL_TEXTURE_2D, _bloomBlurTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 1, 1,
                 0, GL_RGB, GL_UNSIGNED_INT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &_bloomFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _bloomFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _bloomBaseTex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _bloomDpt);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    // Backdrop
    GLfloat backTriangle[] = {
        -1.0f, -1.0f,
         3.0f, -1.0f,
        -1.0f,  3.0f
    };

    Image& filterTex =  getImageBank().getImage("resources/textures/Filter.png");
    _filterTex = GlToolkit::genTextureId(filterTex);
    _filterWidth = filterTex.width();
    _filterHeight = filterTex.height();

    glGenVertexArrays(1, &_fullscreenVao);
    glBindVertexArray(_fullscreenVao);

    glGenBuffers(1, &_fullscreenVbo);
    glBindBuffer(GL_ARRAY_BUFFER, _fullscreenVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(backTriangle), backTriangle, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

void MidEndRenderer::setupShaders()
{
    // Compile shaders
    _gradientShader.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/Filter.vert");
    _gradientShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/Gradient.frag");
    _gradientShader.link();
    _gradientShader.pushProgram();
    _gradientShader.setInt("Filter", 1);
    _gradientShader.setVec2f("TexScale", glm::vec2(1.0f));
    _gradientShader.popProgram();

    _shadowShader.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/Shadow.vert");
    _shadowShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/Shadow.frag");
    _shadowShader.link();
    _shadowShader.pushProgram();
    _shadowShader.popProgram();

    _litShader.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/LitMesh.vert");
    _litShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/LitMesh.frag");
    _litShader.link();
    _litShader.pushProgram();
    _litShader.setInt("DepthTex", 0);
    _litShader.popProgram();

    _unlitShader.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/UnlitMesh.vert");
    _unlitShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/UnlitMesh.frag");
    _unlitShader.link();
    _unlitShader.pushProgram();
    _unlitShader.popProgram();

    _bloomBlurShader.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/Bloom.vert");
    _bloomBlurShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/BloomBlur.frag");
    _bloomBlurShader.link();
    _bloomBlurShader.pushProgram();
    _bloomBlurShader.setInt("BloomBase", 2);
    _bloomBlurShader.popProgram();

    _bloomBlendShader.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/Bloom.vert");
    _bloomBlendShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/BloomBlend.frag");
    _bloomBlendShader.link();
    _bloomBlendShader.pushProgram();
    _bloomBlendShader.setInt("BloomBase", 2);
    _bloomBlendShader.setInt("BloomBlur", 3);
    _bloomBlendShader.popProgram();

    _screenShader.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/Filter.vert");
    _screenShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/Screen.frag");
    _screenShader.link();
    _screenShader.pushProgram();
    _screenShader.setInt("Base", 2);
    _screenShader.setInt("Filter", 1);
    _screenShader.setVec2f("TexScale", glm::vec2(1.0f));
    _screenShader.popProgram();

    _brushShader.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/Filter.vert");
    _brushShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/Brush.frag");
    _brushShader.link();
    _brushShader.pushProgram();
    _brushShader.setInt("Base", 3);
    _brushShader.setInt("Filter", 1);
    _brushShader.setVec2f("TexScale", glm::vec2(1.0f));
    _brushShader.popProgram();

    _grainShader.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/Filter.vert");
    _grainShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/Grain.frag");
    _grainShader.link();
    _grainShader.pushProgram();
    _grainShader.setInt("Base", 2);
    _grainShader.setInt("Filter", 1);
    _grainShader.setVec2f("TexScale", glm::vec2(1.0f));
    _grainShader.popProgram();


    // Set shadow projection view matrix
    _shadowProj = glm::ortho(
            -2.0f, 2.0f,
            -2.0f, 2.0f,
            -2.0f, 2.0f);
}

void MidEndRenderer::render()
{
    int viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, _bloomBlurTex);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, _bloomBaseTex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _filterTex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _shadowTex);

    if(_lightingEnabled)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, _bloomFbo);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             _bloomBaseTex,  0);
        glClear(GL_DEPTH_BUFFER_BIT);
    }

    // Render background
    glBindVertexArray(_fullscreenVao);
    _gradientShader.pushProgram();
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    _gradientShader.popProgram();
    glBindVertexArray(_meshVao);


    // Render shadow map
    if(_lightingEnabled)
    {
        if(_updateShadow)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, _shadowFbo);
            glViewport(0, 0, _shadowSize.x, _shadowSize.y);

            glClearColor(1.0, 1.0, 1.0, 1.0);
            glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

            _shadowShader.pushProgram();
            glDisableVertexAttribArray(1);
            glDisableVertexAttribArray(2);
            glDisableVertexAttribArray(3);
            glDrawArrays(GL_TRIANGLES, 0, _buffElemCount);
            glEnableVertexAttribArray(1);
            glEnableVertexAttribArray(2);
            glEnableVertexAttribArray(3);
            _shadowShader.popProgram();

            glBindFramebuffer(GL_FRAMEBUFFER, _bloomFbo);
            glViewport(viewport[0], viewport[1],
                       viewport[2], viewport[3]);

            glGenerateMipmap(GL_TEXTURE_2D);
            _updateShadow = false;
        }

        _litShader.pushProgram();
    }
    else
    {
        _unlitShader.pushProgram();
    }



    // Render mesh
    glBindVertexArray(_meshVao);
    glDrawArrays(GL_TRIANGLES, 0, _buffElemCount);
    GlProgram::popProgram();


    if(_lightingEnabled)
    {
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        glBindVertexArray(_fullscreenVao);


        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             _bloomBlurTex,  0);
        _bloomBlurShader.pushProgram();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        _bloomBlurShader.popProgram();


        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             _bloomBaseTex,  0);
        _bloomBlendShader.pushProgram();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        _bloomBlendShader.popProgram();


        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             _bloomBlurTex,  0);
        _screenShader.pushProgram();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        _screenShader.popProgram();


        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             _bloomBaseTex,  0);
        _brushShader.pushProgram();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        _brushShader.popProgram();


        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        _grainShader.pushProgram();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        _grainShader.popProgram();


        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
    }

    glBindVertexArray(0);
}
