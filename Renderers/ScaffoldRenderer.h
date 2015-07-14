#ifndef GPUMESH_SCIENTIFICRENDERER
#define GPUMESH_SCIENTIFICRENDERER

#include <GL3/gl3w.h>

#include <CellarWorkbench/GL/GlProgram.h>

#include "AbstractRenderer.h"


class ScaffoldRenderer : public AbstractRenderer
{
public:
    ScaffoldRenderer();
    virtual ~ScaffoldRenderer();

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
    virtual void updateGeometry(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator) override;
    virtual void compileVerts(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator,
            std::vector<float>& verts,
            std::vector<GLubyte>& quals);
    virtual void compileEdges(
            const Mesh& mesh,
            std::vector<GLuint>& edges);

    virtual void clearResources() override;
    virtual void resetResources() override;
    virtual void setupShaders() override;
    virtual void render() override;

    virtual void useWireframeShading();
    virtual void useDiffuseShading();
    virtual void usePhongShading();

    cellar::GlProgram _scaffoldJointProgram;
    cellar::GlProgram _scaffoldTubeProgram;
    cellar::GlProgram _wireframeProgram;

    glm::mat4 _projMat;
    glm::mat4 _viewMat;

    int _vertElemCount;
    int _indxElemCount;

    GLuint _vao;
    GLuint _vbo;
    GLuint _qbo;
    GLuint _ibo;

    int _lightMode;
    float _tubeRadius;
    float _jointRadius;
    float _jointTubeMinRatio;
};

#endif // GPUMESH_MIDENDRENDERER
