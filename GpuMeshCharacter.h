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

    void enterStage() override;
    void beginStep(const scaena::StageTime &time) override;
    void draw(const std::shared_ptr<scaena::View> &view,
              const scaena::StageTime &time) override;
    void exitStage() override;

protected:
    // CPU pipleine
    virtual void resetCpuPipeline();
    virtual void processCpuPipeline();
    virtual void genBoundaryMeshesCpu();
    virtual void triangulateDomainCpu();
    virtual void smoothMeshCpu();

    // GPU pipeline
    virtual void resetGpuPipeline();
    virtual void processGpuPipeline();


    virtual void printStep(int step, const std::string& stepName);
    virtual void moveCamera(double azimuth, double altitude, double distance);
    virtual void moveCutPlane(double azimuth, double altitude, double distance);
    virtual void setupShaders();
    virtual void updateBuffers();

protected:
    cellar::GlProgram _shader;
    int _buffElemCount;
    GLuint _vao;
    GLuint _vbo;
    GLuint _nbo;
    GLuint _ebo;
    GLuint _qbo;

    double _camAzimuth;
    double _camAltitude;
    double _camDistance;

    double _cutAzimuth;
    double _cutAltitude;
    double _cutDistance;

    glm::dmat4 _projection;
    glm::dmat4 _viewMatrix;

    Mesh _mesh;
    int _internalVertices;

private:
    bool _useGpuPipeline;
    bool _processFinished;
    int _stepId;
};

#endif //GpuMesh_CHARACTER
