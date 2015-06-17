#ifndef GpuMesh_CHARACTER
#define GpuMesh_CHARACTER

#include <GL3/gl3w.h>

#include <CellarWorkbench/GL/GlProgram.h>
#include <CellarWorkbench/Camera/Camera.h>
#include <CellarWorkbench/DesignPattern/SpecificObserver.h>

#include <Scaena/Play/Character.h>

#include "DataStructures/Mesh.h"
#include "Meshers/AbstractMesher.h"


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
    cellar::GlProgram _backdropShader;
    cellar::GlProgram _bloomBlurShader;
    cellar::GlProgram _bloomBlendShader;
    int _buffElemCount;
    GLuint _meshVao;
    GLuint _vbo;
    GLuint _nbo;
    GLuint _ebo;
    GLuint _qbo;

    bool _useBackdrop;
    int _backdropWidth;
    int _backdropHeight;
    GLuint _fullscreenVao;
    GLuint _fullscreenVbo;
    GLuint _fullscreenTex;

    bool _lightingEnabled;
    bool _updateShadow;
    glm::mat4 _shadowProj;
    glm::ivec2 _shadowSize;
    GLuint _shadowFbo;
    GLuint _shadowDpt;
    GLuint _shadowTex;
    GLuint _bloomBaseFbo;
    GLuint _bloomBlurFbo;
    GLuint _bloomBaseTex;
    GLuint _bloomBlurTex;
    GLuint _bloomDpt;

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


private:
    std::unique_ptr<AbstractMesher> _mesher;

    static const glm::vec3 nullVec;
    static const glm::vec3 upVec;
};

#endif //GpuMesh_CHARACTER
