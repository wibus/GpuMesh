#include "GpuLaplacianSmoother.h"

#include <iostream>
#include <chrono>

using namespace std;




GpuLaplacianSmoother::GpuLaplacianSmoother(
        double moveFactor,
        double gainThreshold) :
    AbstractSmoother(moveFactor, gainThreshold),
    _initialized(false)
{

}

GpuLaplacianSmoother::~GpuLaplacianSmoother()
{

}

void GpuLaplacianSmoother::smoothMesh(Mesh& mesh, AbstractEvaluator& evaluator)
{
    GLuint vertSsbo = mesh.glBuffer(EMeshBuffer::VERT);
    GLuint topoSsbo = mesh.glBuffer(EMeshBuffer::TOPO);
    GLuint neigSsbo = mesh.glBuffer(EMeshBuffer::NEIG);

    if(!_initialized)
    {
        initializeProgram(mesh);

        _initialized = true;
    }
    else
    {
        // Absurdly make subsequent passes much more faster...
        // I guess it's because the driver put buffer back on GPU.
        // It looks like glGetBufferSubData take it out of the GPU.
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertSsbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, _vertTmpBuffSize,
                     _vertTmpBuff.data(),      GL_STATIC_DRAW);

    }


    int vertCount = mesh.vertCount();

    _smoothingProgram.pushProgram();
    _smoothingProgram.setInt("VertCount", vertCount);
    _smoothingProgram.setFloat("MoveCoeff", _moveFactor);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, topoSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, neigSsbo);

    auto tStart = chrono::high_resolution_clock::now();
    auto tMiddle = tStart;
    auto tEnd = tStart;

    evaluateInitialMeshQuality(mesh, evaluator);
    while(evaluateIterationMeshQuality(mesh, evaluator))
    {
        glDispatchCompute(ceil(vertCount / 256.0), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


        if(_smoothPassId == 100)
        {
            tMiddle = chrono::high_resolution_clock::now();

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertSsbo);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                               _vertTmpBuffSize,         _vertTmpBuff.data());
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

            tEnd = chrono::high_resolution_clock::now();

            for(int i=0; i < vertCount; ++i)
            {
                mesh.vert[i].p = glm::dvec3(_vertTmpBuff[i]);
            }
        }
    }

    _smoothingProgram.popProgram();


    chrono::microseconds dtMid;
    dtMid = chrono::duration_cast<chrono::microseconds>(tMiddle - tStart);
    cout << "Total CS time = " << dtMid.count() / 1000.0 << "ms" << endl;
    chrono::microseconds dtEnd;
    dtEnd = chrono::duration_cast<chrono::microseconds>(tEnd - tMiddle);
    cout << "Get buffer time = " << dtEnd.count() / 1000.0 << "ms" << endl;
}

void GpuLaplacianSmoother::initializeProgram(Mesh& mesh)
{
    cout << "Initializing Laplacian smoothing compute shader" << endl;
    _smoothingProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/LaplacianSmoothing.glsl");
    _smoothingProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/ElbowPipeBoundaries.glsl");
    _smoothingProgram.link();

    _vertTmpBuff.resize(mesh.vertCount());
    _vertTmpBuffSize = sizeof(decltype(_vertTmpBuff.front())) * _vertTmpBuff.size();
}
