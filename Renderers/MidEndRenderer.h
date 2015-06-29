#ifndef GPUMESH_MIDENDRENDERER
#define GPUMESH_MIDENDRENDERER

#include <GL3/gl3w.h>

#include <CellarWorkbench/GL/GlProgram.h>

#include "AbstractRenderer.h"


class MidEndRenderer : public AbstractRenderer
{
public:
    MidEndRenderer();
    virtual ~MidEndRenderer();

    virtual void notify(cellar::CameraMsg& msg) override;

    virtual void updateCamera(const glm::mat4& view,
                              const glm::vec3& pos) override;
    virtual void updateLight(const glm::mat4& view,
                             const glm::vec3& pos) override;
    virtual void updateCutPlane(const glm::dvec4& cutEq) override;

    virtual void handleKeyPress(const scaena::KeyboardEvent& event) override;
    virtual void handleInputs(const scaena::SynchronousKeyboard& keyboard,
                              const scaena::SynchronousMouse& mouse) override;



protected:
    virtual void updateGeometry(const Mesh& mesh) override;

    virtual void clearResources() override;
    virtual void resetResources() override;
    virtual void setupShaders() override;
    virtual void render() override;

    cellar::GlProgram _litShader;
    cellar::GlProgram _unlitShader;
    cellar::GlProgram _shadowShader;
    cellar::GlProgram _bloomBlurShader;
    cellar::GlProgram _bloomBlendShader;
    cellar::GlProgram _gradientShader;
    cellar::GlProgram _screenShader;
    cellar::GlProgram _brushShader;
    cellar::GlProgram _grainShader;
    int _buffElemCount;
    GLuint _meshVao;
    GLuint _vbo;
    GLuint _nbo;
    GLuint _ebo;
    GLuint _qbo;


    bool _useBackdrop;
    int _filterWidth;
    int _filterHeight;
    GLuint _fullscreenVao;
    GLuint _fullscreenVbo;
    GLuint _filterTex;


    bool _lightingEnabled;
    bool _updateShadow;
    glm::mat4 _shadowProj;
    glm::ivec2 _shadowSize;
    GLuint _shadowFbo;
    GLuint _shadowDpt;
    GLuint _shadowTex;
    GLuint _bloomFbo;
    GLuint _bloomDpt;
    GLuint _bloomBaseTex;
    GLuint _bloomBlurTex;


    bool _isPhysicalCut;
    glm::dvec4 _cutPlane;
    std::unique_ptr<AbstractEvaluator> _evaluator;
};

#endif // GPUMESH_MIDENDRENDERER
