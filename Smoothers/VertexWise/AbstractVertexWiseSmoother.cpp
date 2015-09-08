#include "AbstractVertexWiseSmoother.h"

#include <thread>
#include <atomic>
#include <condition_variable>
#include <algorithm>

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
    size_t vertCount = mesh.verts.size();
    std::vector<uint> vIds(vertCount);
    std::iota(std::begin(vIds), std::end(vIds), 0);

    _smoothPassId = 0;
    while(evaluateMeshQualitySerial(mesh, evaluator))
    {
        smoothVertices(mesh, evaluator, vIds);
    }

    mesh.updateGpuVertices();
}

void AbstractVertexWiseSmoother::smoothMeshThread(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    // TODO : Use a thread pool    
    size_t groupCount = mesh.exclusiveGroups.size();
    uint threadCount = thread::hardware_concurrency();

    std::mutex mutex;
    std::condition_variable cv;
    std::atomic_int done( 0 );
    std::atomic_int step( 0 );

    _smoothPassId = 0;
    while(evaluateMeshQualityThread(mesh, evaluator))
    {
        vector<thread> workers;
        for(uint t=0; t < threadCount; ++t)
        {
            workers.push_back(thread([&, t]() {
                for(size_t g=0; g < groupCount; ++g)
                {
                    const std::vector<uint>& group =
                            mesh.exclusiveGroups[g];

                    size_t groupSize = group.size();
                    std::vector<uint> vIds(
                        group.begin() + (groupSize * t) / threadCount,
                        group.begin() + (groupSize * (t+1)) / threadCount);

                    smoothVertices(mesh, evaluator, vIds);

                    if(g < groupCount-1)
                    {
                        std::unique_lock<std::mutex> lk(mutex);
                        if(done.fetch_add( 1 ) == threadCount-1)
                        {
                            ++step;
                            done.store( 0 );
                            cv.notify_all();
                        }
                        else
                        {
                            cv.wait(lk, [&](){ return step > g; });
                        }
                    }
                }
            }));
        }

        for(uint t=0; t < threadCount; ++t)
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


    vector<ExclusiveDispatch> dispatches;
    organizeDispatches(mesh, WORKGROUP_SIZE, dispatches);
    size_t dispatchCount = dispatches.size();

    _vertSmoothProgram.pushProgram();
    setVertexProgramUniforms(mesh, _vertSmoothProgram);
    _vertSmoothProgram.popProgram();

    _smoothPassId = 0;
    while(evaluateMeshQualityGlsl(mesh, evaluator))
    {
        mesh.bindShaderStorageBuffers();

        _vertSmoothProgram.pushProgram();

        for(size_t d=0; d < dispatchCount; ++d)
        {
            const ExclusiveDispatch& dispatch = dispatches[d];
            _vertSmoothProgram.setInt("GroupBase", dispatch.base);
            _vertSmoothProgram.setInt("GroupSize", dispatch.size);

            glDispatchCompute(dispatch.workgroupCount, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        _vertSmoothProgram.popProgram();
    }


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
        "Initializing smoothing compute shader", "AbstractVertexWiseSmoother"));

    _modelBoundsShader = mesh.modelBoundsShaderName();
    _shapeMeasureShader = evaluator.shapeMeasureShader();

    _vertSmoothProgram.clearShaders();
    _vertSmoothProgram.addShader(GL_COMPUTE_SHADER,
        _modelBoundsShader);
    _vertSmoothProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        _shapeMeasureShader.c_str()});
    _vertSmoothProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Quality/QualityInterface.glsl"});
    _vertSmoothProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        SmoothingHelper::shaderName().c_str()});
    _vertSmoothProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Smoothing/VertexWise/SmoothVertices.glsl"});
    for(const string& shader : _smoothShaders)
    {
        _vertSmoothProgram.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            shader.c_str()});
    }
    _vertSmoothProgram.link();

    mesh.uploadGeometry(_vertSmoothProgram);


    _initialized = true;
}

void AbstractVertexWiseSmoother::setVertexProgramUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program)
{
    program.setFloat("MoveCoeff", _moveFactor);
}
