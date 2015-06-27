#ifndef GpuMesh_CHARACTER
#define GpuMesh_CHARACTER

#include <GL3/gl3w.h>

#include <CellarWorkbench/GL/GlProgram.h>
#include <CellarWorkbench/Camera/Camera.h>
#include <CellarWorkbench/DesignPattern/SpecificObserver.h>

#include <PropRoom2D/Prop/Hud/TextHud.h>

#include <Scaena/Play/Character.h>

#include "DataStructures/Mesh.h"
#include "Meshers/AbstractMesher.h"
#include "Smoothers/AbstractSmoother.h"


class GpuMeshCharacter :
        public scaena::Character,
        public cellar::SpecificObserver<cellar::CameraMsg>
{
public:
    GpuMeshCharacter();

    virtual void enterStage() override;
    virtual void beginStep(const scaena::StageTime &time) override;
    virtual void draw(const std::shared_ptr<scaena::View> &view,
                      const scaena::StageTime &time) override;
    virtual void exitStage() override;

    virtual bool keyPressEvent(const scaena::KeyboardEvent &event) override;

    virtual void notify(cellar::CameraMsg& msg) override;

protected:
    virtual void resetPipeline();
    virtual void processPipeline();
    virtual void scheduleSmoothing();
    virtual void printStep(int step, const std::string& stepName);

    virtual void moveCamera(float azimuth, float altitude, float distance);
    virtual void moveCutPlane(float azimuth, float altitude, float distance);
    virtual void moveLight(float azimuth, float altitude, float distance);

    virtual void setupShaders();
    virtual void updateBuffers();
    virtual void resetResources();

protected:
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

    glm::mat4 _camProj;
    float _camAzimuth;
    float _camAltitude;
    float _camDistance;

    bool _isPhysicalCut;
    float _cutAzimuth;
    float _cutAltitude;
    float _cutDistance;
    glm::dvec4 _physicalCutPlaneEq;

    float _lightAzimuth;
    float _lightAltitude;
    float _lightDistance;

    Mesh _mesh;
    int _stepId;
    bool _processFinished;
    bool _mustUpdateBuffers;


private:
    std::unique_ptr<AbstractMesher> _mesher;
    std::unique_ptr<AbstractSmoother> _smoother;

    static const glm::vec3 nullVec;
    static const glm::vec3 upVec;

    std::shared_ptr<prop2::TextHud> _fps;
    std::shared_ptr<prop2::TextHud> _ups;
};

#endif //GpuMesh_CHARACTER
