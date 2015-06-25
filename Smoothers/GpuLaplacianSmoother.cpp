#include "GpuLaplacianSmoother.h"

#include <iostream>

using namespace std;


GpuLaplacianSmoother::GpuLaplacianSmoother(
        Mesh &mesh,
        double moveFactor,
        double gainThreshold) :
    AbstractSmoother(mesh, moveFactor, gainThreshold),
    _initialized(false),
    _topologyChanged(true)
{

}

GpuLaplacianSmoother::~GpuLaplacianSmoother()
{

}

void GpuLaplacianSmoother::smoothMesh()
{
    if(!_initialized)
    {
        initializeProgram();

        _initialized = true;
    }

    if(_topologyChanged)
    {
        updateBuffers();

        _topologyChanged = false;
    }

    int vertCount = _mesh.vertCount();

    _smoothingProgram.pushProgram();
    _smoothingProgram.setInt("VertCount", vertCount);
    _smoothingProgram.setFloat("MoveCoeff", _moveFactor);
    glDispatchCompute(ceil(vertCount / 128.0), 1, 1);
    _smoothingProgram.popProgram();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    glm::vec4* vertPos = (glm::vec4*) glMapBuffer(
            GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    for(int i=0; i<vertCount; ++i)
    {
        _mesh.vert[i].p = glm::dvec3(vertPos[i]);
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

void GpuLaplacianSmoother::initializeProgram()
{
    cout << "Initializing Laplacian smoothing compute shader" << endl;
    _smoothingProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/LaplacianSmoothing.glsl");
    _smoothingProgram.link();

    glGenBuffers(1, &_vertSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _vertSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GpuLaplacianSmoother::updateBuffers()
{
    cout << "Updating shader storage buffers" << endl;

    int vertCount = _mesh.vert.size();
    glm::vec4* vertPos = new glm::vec4[vertCount];
    for(int i=0; i<vertCount; ++i)
    {
        vertPos[i] = glm::vec4(_mesh.vert[i].p, 0.0);
    }

    size_t vertSize = sizeof(glm::vec4) * vertCount;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, vertSize, vertPos, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    delete[] vertPos;
}
