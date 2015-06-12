#ifndef GpuMesh_CHARACTER
#define GpuMesh_CHARACTER

#include <GL3/gl3w.h>

#include <CellarWorkbench/GL/GlProgram.h>

#include <Scaena/Play/Character.h>

#include "Mesh.h"


class GpuMeshCharacter : public scaena::Character
{
public:
    GpuMeshCharacter();

    virtual void enterStage() override;
    virtual void beginStep(const scaena::StageTime &time) override;
    virtual void draw(const std::shared_ptr<scaena::View> &view,
                      const scaena::StageTime &time) override;
    virtual void exitStage() override;

    virtual bool keyPressEvent(const scaena::KeyboardEvent &event) override;

protected:
    // CPU pipleine
    virtual void resetCpuPipeline();
    virtual void processCpuPipeline();
    virtual void genBoundaryMeshesCpu();
    virtual void triangulateDomainCpu();
    virtual void computeAdjacencyCpu();
    virtual void smoothMeshCpu();

    // GPU pipeline
    virtual void resetGpuPipeline();
    virtual void processGpuPipeline();


    virtual void printStep(int step, const std::string& stepName);
    virtual void moveCamera(float azimuth, float altitude, float distance);
    virtual void moveCutPlane(float azimuth, float altitude, float distance);
    virtual void moveLight(float azimuth, float altitude, float distance);
    virtual void setupShaders();
    virtual void updateBuffers();

protected:
    cellar::GlProgram _litShader;
    cellar::GlProgram _unlitShader;
    int _buffElemCount;
    GLuint _vao;
    GLuint _vbo;
    GLuint _nbo;
    GLuint _ebo;
    GLuint _qbo;

    bool _useLitShader;
    bool _shadowEnabled;
    bool _updateShadow;
    glm::mat4 _shadowProj;
    glm::ivec2 _shadowSize;
    cellar::GlProgram _shadowShader;
    GLuint _shadowFbo;
    GLuint _shadowDpt;
    GLuint _shadowTex;

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
    int _internalVertices;

private:
    bool _useGpuPipeline;
    bool _processFinished;
    int _stepId;

    static const glm::vec3 nullVec;
    static const glm::vec3 upVec;
};

#endif //GpuMesh_CHARACTER
