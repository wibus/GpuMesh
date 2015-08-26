#include "AbstractVertexWiseSmoother.h"

#include <thread>

#include "../SmoothingHelper.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;
using namespace cellar;


const size_t AbstractVertexWiseSmoother::WORKGROUP_SIZE = 256;

AbstractVertexWiseSmoother::AbstractVertexWiseSmoother(
        const std::vector<std::string>& smoothShaders) :
    _initialized(false),
    _smoothShaders(smoothShaders)
{

}

AbstractVertexWiseSmoother::~AbstractVertexWiseSmoother()
{

}

void AbstractVertexWiseSmoother::smoothMeshSerial(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    _smoothPassId = 0;
    size_t vertCount = mesh.verts.size();
    while(evaluateMeshQualitySerial(mesh, evaluator))
    {
        smoothVertices(mesh, evaluator, 0, vertCount, false);
    }

    mesh.updateGpuVertices();
}

void AbstractVertexWiseSmoother::smoothMeshThread(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    // TODO : Use a thread pool

    _smoothPassId = 0;
    size_t vertCount = mesh.verts.size();
    while(evaluateMeshQualityThread(mesh, evaluator))
    {
        vector<thread> workers;
        uint coreCountHint = thread::hardware_concurrency();
        for(uint t=0; t < coreCountHint; ++t)
        {
            workers.push_back(thread([&, t]() {
                size_t first = (vertCount * t) / coreCountHint;
                size_t last = (vertCount * (t+1)) / coreCountHint;
                smoothVertices(mesh, evaluator, first, last, true);
            }));
        }

        for(uint t=0; t < coreCountHint; ++t)
        {
            workers[t].join();
        }
    }

    mesh.updateGpuVertices();
}

void AbstractVertexWiseSmoother::smoothMeshGlsl(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    initializeProgram(mesh, evaluator);

    // There's no need to upload vertices again, but absurdly
    // this makes subsequent passes much more faster...
    // I guess it's because the driver put buffer back on GPU.
    // It looks like glGetBufferSubData takes it out of the GPU.
    mesh.updateGpuVertices();


    _smoothPassId = 0;
    _smoothingProgram.pushProgram();
    _smoothingProgram.setFloat("MoveCoeff", _moveFactor);
    mesh.bindShaderStorageBuffers();
    const int vertCount = mesh.vertCount();
    while(evaluateMeshQualityGlsl(mesh, evaluator))
    {
        glDispatchCompute(glm::ceil(vertCount / double(WORKGROUP_SIZE)), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    _smoothingProgram.popProgram();


    // Fetch new vertices' position
    mesh.updateCpuVertices();
}

void AbstractVertexWiseSmoother::initializeProgram(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    if(_initialized &&
       _modelBoundsShader == mesh.modelBoundsShaderName() &&
       _shapeMeasureShader == evaluator.shapeMeasureShader())
        return;


    getLog().postMessage(new Message('I', false,
        "Initializing smoothing compute shader", "AbstractSmoother"));

    _modelBoundsShader = mesh.modelBoundsShaderName();
    _shapeMeasureShader = evaluator.shapeMeasureShader();

    _smoothingProgram.clearShaders();
    _smoothingProgram.addShader(GL_COMPUTE_SHADER,
        _modelBoundsShader);
    _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        _shapeMeasureShader.c_str()});
    _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Quality/QualityInterface.glsl"});
    _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        SmoothingHelper::shaderName().c_str()});
    for(const string& shader : _smoothShaders)
    {
        _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            shader.c_str()});
    }
    _smoothingProgram.link();

    mesh.uploadGeometry(_smoothingProgram);


    _initialized = true;
}
