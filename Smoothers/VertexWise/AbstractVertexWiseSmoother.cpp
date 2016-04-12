#include "AbstractVertexWiseSmoother.h"

#include <thread>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <chrono>

#include "DataStructures/MeshCrew.h"
#include "Samplers/AbstractSampler.h"
#include "Measurers/AbstractMeasurer.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Topologists/AbstractTopologist.h"

using namespace std;
using namespace cellar;


// CUDA Drivers
extern bool verboseCuda;
void smoothCudaVertices(
        const IndependentDispatch& dispatch,
        size_t workgroupSize,
        float moveCoeff);


const size_t AbstractVertexWiseSmoother::WORKGROUP_SIZE = 256;

AbstractVertexWiseSmoother::AbstractVertexWiseSmoother(
        const std::vector<std::string>& smoothShaders,
        const installCudaFct installCuda) :
    AbstractSmoother(installCuda),
    _initialized(false),
    _smoothShaders(smoothShaders)
{

}

AbstractVertexWiseSmoother::~AbstractVertexWiseSmoother()
{

}

void AbstractVertexWiseSmoother::smoothMeshSerial(
        Mesh& mesh,
        const MeshCrew& crew)
{
    size_t vertCount = mesh.verts.size();
    std::vector<uint> vIds(vertCount);
    std::iota(std::begin(vIds), std::end(vIds), 0);

    _smoothPassId = 0;
    while(evaluateMeshQualitySerial(mesh, crew))
    {
        if(crew.needTopologicalModifications(_smoothPassId))
        {
            verboseCuda = false;
            crew.topologist().restructureMesh(mesh, crew);
            verboseCuda = true;
        }

        smoothVertices(mesh, crew, vIds);
    }

    mesh.updateVerticesFromCpu();
}

void AbstractVertexWiseSmoother::smoothMeshThread(
        Mesh& mesh,
        const MeshCrew& crew)
{
    // TODO : Use a thread pool
    uint threadCount = thread::hardware_concurrency();

    std::mutex mutex;
    std::condition_variable cv;

    _smoothPassId = 0;
    while(evaluateMeshQualityThread(mesh, crew))
    {
        if(crew.needTopologicalModifications(_smoothPassId))
        {
            verboseCuda = false;
            crew.topologist().restructureMesh(mesh, crew);
            verboseCuda = true;
        }

        std::atomic<int> done( 0 );
        std::atomic<int> step( 0 );

        vector<thread> workers;
        for(uint t=0; t < threadCount; ++t)
        {
            workers.push_back(thread([&, t]() {
                size_t groupCount = mesh.independentGroups.size();
                for(size_t g=0; g < groupCount; ++g)
                {
                    const std::vector<uint>& group =
                            mesh.independentGroups[g];

                    size_t groupSize = group.size();
                    std::vector<uint> vIds(
                        group.begin() + (groupSize * t) / threadCount,
                        group.begin() + (groupSize * (t+1)) / threadCount);

                    smoothVertices(mesh, crew, vIds);

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

    mesh.updateVerticesFromCpu();
}

void AbstractVertexWiseSmoother::smoothMeshGlsl(
        Mesh& mesh,
        const MeshCrew& crew)
{
    initializeProgram(mesh, crew);

    // There's no need to upload vertices again, but absurdly
    // this makes subsequent passes much more faster...
    // I guess it's because the driver put buffer back on GPU.
    // It looks like glGetBufferSubData takes it out of the GPU.

    // Note (2016-04-04) : This trick doesn't seem to be significant anymore...
    //mesh.updateVerticesFromCpu();


    vector<IndependentDispatch> dispatches;
    organizeDispatches(mesh, WORKGROUP_SIZE, dispatches);
    size_t dispatchCount = dispatches.size();

    _vertSmoothProgram.pushProgram();
    setVertexProgramUniforms(mesh, _vertSmoothProgram);
    _vertSmoothProgram.popProgram();

    _smoothPassId = 0;
    while(evaluateMeshQualityGlsl(mesh, crew))
    {
        if(crew.needTopologicalModifications(_smoothPassId))
        {
            verboseCuda = false;
            mesh.updateVerticesFromGlsl();
            crew.topologist().restructureMesh(mesh, crew);
            mesh.updateVerticesFromCpu();
            mesh.updateGpuTopology();
            verboseCuda = true;

            organizeDispatches(mesh, WORKGROUP_SIZE, dispatches);
            dispatchCount = dispatches.size();
        }


        _vertSmoothProgram.pushProgram();
        mesh.bindShaderStorageBuffers();

        for(size_t d=0; d < dispatchCount; ++d)
        {
            const IndependentDispatch& dispatch = dispatches[d];
            _vertSmoothProgram.setInt("GroupBase", dispatch.base);
            _vertSmoothProgram.setInt("GroupSize", dispatch.size);

            glDispatchCompute(dispatch.workgroupCount, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
        }

        _vertSmoothProgram.popProgram();
    }


    // Fetch new vertices' position
    mesh.updateVerticesFromGlsl();
}

void AbstractVertexWiseSmoother::smoothMeshCuda(
        Mesh& mesh,
        const MeshCrew& crew)
{
    initializeProgram(mesh, crew);
    _installCudaSmoother();

    vector<IndependentDispatch> dispatches;
    organizeDispatches(mesh, WORKGROUP_SIZE, dispatches);
    size_t dispatchCount = dispatches.size();


    _smoothPassId = 0;
    while(evaluateMeshQualityCuda(mesh, crew))
    {
        if(crew.needTopologicalModifications(_smoothPassId))
        {
            verboseCuda = false;
            mesh.updateVerticesFromCuda();
            crew.topologist().restructureMesh(mesh, crew);
            mesh.updateVerticesFromCpu();
            mesh.updateGpuTopology();
            verboseCuda = true;

            organizeDispatches(mesh, WORKGROUP_SIZE, dispatches);
            dispatchCount = dispatches.size();
        }

        for(size_t d=0; d < dispatchCount; ++d)
        {
            const IndependentDispatch& dispatch = dispatches[d];
            smoothCudaVertices(dispatch, WORKGROUP_SIZE, _moveCoeff);
        }
    }


    // Fetch new vertices' position
    mesh.updateVerticesFromCuda();
}

void AbstractVertexWiseSmoother::initializeProgram(
        Mesh& mesh,
        const MeshCrew& crew)
{
    if(_initialized &&
       _modelBoundsShader == mesh.modelBoundsShaderName() &&
       _samplingShader == crew.sampler().samplingShader() &&
       _evaluationShader == crew.evaluator().evaluationShader() &&
       _measureShader == crew.measurer().measureShader())
            return;


    getLog().postMessage(new Message('I', false,
        "Initializing smoothing compute shader", "AbstractVertexWiseSmoother"));

    _modelBoundsShader = mesh.modelBoundsShaderName();
    _samplingShader = crew.sampler().samplingShader();
    _evaluationShader = crew.evaluator().evaluationShader();
    _measureShader = crew.measurer().measureShader();

    _vertSmoothProgram.reset();
    crew.installPlugins(mesh, _vertSmoothProgram);
    _vertSmoothProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        smoothingUtilsShader().c_str()});
    _vertSmoothProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/glsl/compute/Smoothing/VertexWise/SmoothVertices.glsl"});
    for(const string& shader : _smoothShaders)
    {
        _vertSmoothProgram.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            shader.c_str()});
    }

    _vertSmoothProgram.link();
    crew.setPluginUniforms(mesh, _vertSmoothProgram);

    /*
    const GlProgramBinary& binary = _vertSmoothProgram.getBinary();
    std::ofstream file("Smoother_binary.txt", std::ios_base::trunc);
    if(file.is_open())
    {
        file << "Length: " << binary.length << endl;
        file << "Format: " << binary.format << endl;
        file << "Binary ->" << endl;
        file.write(binary.binary, binary.length);
        file.close();
    }
    */

    _initialized = true;
}

void AbstractVertexWiseSmoother::setVertexProgramUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program)
{
    program.setFloat("MoveCoeff", _moveCoeff);
}

void AbstractVertexWiseSmoother::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{
    plot.addSmoothingProperty("Category", "Vertex-Wise");
}
