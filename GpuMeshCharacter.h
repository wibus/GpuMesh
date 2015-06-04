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
    virtual void printStep(int step, const std::string& stepName);
    virtual void moveCamera(double azimuth, double altitude, double distance);
    virtual void setupShaders();
    virtual void updateBuffers();

    virtual void resetCpuPipeline();
    virtual void processCpuPipeline();
    virtual void genBoundaryMeshesCpu();
    virtual void triangulateDomainCpu();
    virtual void smoothMeshCpu();

    virtual void resetGpuPipeline();
    virtual void processGpuPipeline();

protected:
    cellar::GlProgram _shader;
    GLuint _vao;
    GLuint _vbo;
    GLuint _ibo;

    double _azimuth;
    double _altitude;
    double _distance;
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
