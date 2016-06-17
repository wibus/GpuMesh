#include "AbstractVertexWiseSmoother.h"

#include <thread>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <chrono>

#include "Boundaries/AbstractBoundary.h"
#include "DataStructures/GpuMesh.h"
#include "DataStructures/MeshCrew.h"
#include "DataStructures/NodeGroups.h"
#include "Samplers/AbstractSampler.h"
#include "Measurers/AbstractMeasurer.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Topologists/AbstractTopologist.h"

using namespace std;
using namespace cellar;


// CUDA Drivers
extern bool verboseCuda;
void smoothCudaVertices(
        const NodeGroups::GpuDispatch& dispatch,
        size_t workgroupSize,
        float moveCoeff);
void fetchCudaSubsurfaceVertices(
        std::vector<MeshVert>& verts,
        const NodeGroups::ParallelGroup& group);
void sendCudaBoundaryVertices(
        const std::vector<MeshVert>& verts,
        const NodeGroups::ParallelGroup& group);


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
    bool isTopoEnabled = crew.topologist()
            .needTopologicalModifications(mesh);

    _smoothPassId = INITIAL_PASS_ID;
    while(evaluateMeshQualitySerial(mesh, crew))
    {
        if(isTopoEnabled)
        {
            verboseCuda = false;
            crew.topologist().restructureMesh(mesh, crew);
            verboseCuda = true;
        }

        while(evaluateMeshQualitySerial(mesh, crew))
        {
            smoothVertices(mesh, crew,
                mesh.nodeGroups().serialGroup());
        }

        if(isTopoEnabled)
            _smoothPassId = COMPARE_PASS_ID;
        else
            break;
    }
}

void AbstractVertexWiseSmoother::smoothMeshThread(
        Mesh& mesh,
        const MeshCrew& crew)
{
    bool isTopoEnabled = crew.topologist()
            .needTopologicalModifications(mesh);

    // TODO : Use a thread pool
    uint threadCount = thread::hardware_concurrency();    
    mesh.nodeGroups().setCpuWorkgroupSize(threadCount);

    _smoothPassId = INITIAL_PASS_ID;
    while(evaluateMeshQualityThread(mesh, crew))
    {
        if(isTopoEnabled)
        {
            verboseCuda = false;
            crew.topologist().restructureMesh(mesh, crew);
            verboseCuda = true;
        }

        while(evaluateMeshQualityThread(mesh, crew))
        {
            std::mutex mutex;
            std::condition_variable cv;
            std::atomic<int> done( 0 );
            std::atomic<int> step( 0 );

            vector<thread> workers;
            for(uint t=0; t < threadCount; ++t)
            {
                workers.push_back(thread([&, t]() {
                    size_t groupCount = mesh.nodeGroups().count();
                    for(size_t g=0; g < groupCount; ++g)
                    {
                        smoothVertices(mesh, crew,
                            mesh.nodeGroups().parallelGroups()[g].allDispatchedNodes[t]);

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

        if(isTopoEnabled)
            _smoothPassId = COMPARE_PASS_ID;
        else
            break;
    }
}

void AbstractVertexWiseSmoother::smoothMeshGlsl(
        Mesh& mesh,
        const MeshCrew& crew)
{
    initializeProgram(mesh, crew);

    mesh.updateGlslTopology();
    mesh.updateGlslVertices();

    _vertSmoothProgram.pushProgram();
    crew.setPluginGlslUniforms(mesh, _vertSmoothProgram);
    setVertexProgramUniforms(mesh, _vertSmoothProgram);
    _vertSmoothProgram.popProgram();

    // There's no need to upload vertices again, but absurdly
    // this makes subsequent passes much more faster...
    // I guess it's because the driver put buffer back on GPU.
    // It looks like glGetBufferSubData takes it out of the GPU.

    // Note (2016-04-04) : This trick doesn't seem to be significant anymore...
    //mesh.updateVerticesFromCpu();

    size_t groupCount = mesh.nodeGroups().count();
    uint threadCount = thread::hardware_concurrency();
    mesh.nodeGroups().setCpuWorkgroupSize(threadCount);
    mesh.nodeGroups().setGpuWorkgroupSize(WORKGROUP_SIZE);

    bool isTopoEnabled = crew.topologist()
            .needTopologicalModifications(mesh);

    _smoothPassId = INITIAL_PASS_ID;
    while(evaluateMeshQualityGlsl(mesh, crew))
    {
        if(isTopoEnabled)
        {
            verboseCuda = false;
            mesh.fetchGlslVertices();
            crew.topologist().restructureMesh(mesh, crew);
            mesh.updateGlslTopology();
            mesh.updateGlslVertices();
            verboseCuda = true;

            groupCount = mesh.nodeGroups().count();
        }

        while(evaluateMeshQualityGlsl(mesh, crew))
        {
            std::mutex mutex;
            std::condition_variable groupCv;
            std::condition_variable memcpyCv;
            std::atomic<int> moveDone( 0 );
            std::atomic<int> stepDone( 0 );
            std::atomic<bool> memcpyDone( false );

            vector<thread> workers;
            for(uint t=0; t < threadCount; ++t)
            {
                workers.push_back(thread([&, t]() {
                    for(size_t g=0; g < groupCount; ++g)
                    {
                        const NodeGroups::ParallelGroup& group =
                            mesh.nodeGroups().parallelGroups()[g];

                        smoothVertices(mesh, crew,
                            group.cpuOnlyDispatchedNodes[t]);

                        if(g < groupCount)
                        {
                            std::unique_lock<std::mutex> lk(mutex);
                            if(moveDone.fetch_add( 1 ) == threadCount)
                            {
                                ++stepDone;
                                moveDone.store( 0 );
                                memcpyDone = false;
                                groupCv.notify_all();
                            }
                            else
                            {
                                groupCv.wait(lk, [&](){ return stepDone > g; });
                            }

                            memcpyCv.wait(lk, [&](){ return memcpyDone.load(); });
                        }
                    }
                }));
            }

            _vertSmoothProgram.pushProgram();
            mesh.bindGlShaderStorageBuffers();
            for(size_t g=0; g < groupCount; ++g)
            {
                const NodeGroups::ParallelGroup& group =
                        mesh.nodeGroups().parallelGroups()[g];

                const NodeGroups::GpuDispatch& dispatch = group.gpuDispatch;
                _vertSmoothProgram.setInt("GroupBase", dispatch.gpuBufferBase);
                _vertSmoothProgram.setInt("GroupSize", dispatch.gpuBufferSize);

                glDispatchCompute(dispatch.workgroupCount, 1, 1);
                glMemoryBarrier(GL_ALL_BARRIER_BITS);


                glBindBuffer(GL_SHADER_STORAGE_BUFFER,
                             mesh.glBuffer(EMeshBuffer::VERT));

                // Fetch subsurface vertex positions from GPU
                size_t subsurfaceSize =
                        group.subsurfaceRange.end -
                        group.subsurfaceRange.begin;

                if(subsurfaceSize > 0)
                {
                    subsurfaceSize *= sizeof(GpuVert);
                    size_t subsurfaceBase = group.subsurfaceRange.begin * sizeof(GpuVert);
                    GpuVert* boundVerts = static_cast<GpuVert*>(
                        glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                            subsurfaceBase, subsurfaceSize,
                            GL_MAP_READ_BIT));

                    for(size_t vId = group.subsurfaceRange.begin, bId=0;
                        vId < group.subsurfaceRange.end; ++vId, ++bId)
                    {
                        MeshVert vert(boundVerts[bId]);
                        mesh.verts[vId] = vert;
                    }

                    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                }


                // Synchronize with CPU workers
                if(g < groupCount)
                {
                    std::unique_lock<std::mutex> lk(mutex);
                    if(moveDone.fetch_add( 1 ) == threadCount)
                    {
                        ++stepDone;
                        moveDone.store( 0 );
                        memcpyDone = false;
                        groupCv.notify_all();
                    }
                    else
                    {
                        groupCv.wait(lk, [&](){ return stepDone > g; });
                    }
                }


                // Send boundary vertex positions to GPU
                size_t boundarySize =
                        group.boundaryRange.end -
                        group.boundaryRange.begin;

                if(boundarySize > 0)
                {
                    boundarySize *= sizeof(GpuVert);
                    size_t boundaryBase = group.boundaryRange.begin * sizeof(GpuVert);
                    GpuVert* boundVerts = static_cast<GpuVert*>(
                        glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                            boundaryBase, boundarySize,
                            GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT));

                    for(size_t vId = group.boundaryRange.begin, bId=0;
                        vId < group.boundaryRange.end; ++vId, ++bId)
                    {
                        GpuVert vert(mesh.verts[vId]);
                        boundVerts[bId] = vert;
                    }

                    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                }

                glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
                glMemoryBarrier(GL_ALL_BARRIER_BITS);


                // Wake-up threads
                mutex.lock();
                memcpyDone = true;
                mutex.unlock();
                memcpyCv.notify_all();
            }
            _vertSmoothProgram.popProgram();

            for(uint t=0; t < threadCount; ++t)
            {
                workers[t].join();
            }
        }

        if(isTopoEnabled)
            _smoothPassId = COMPARE_PASS_ID;
        else
            break;
    }


    // Fetch new vertices' position
    mesh.fetchGlslVertices();
    mesh.clearGlslMemory();
}

void AbstractVertexWiseSmoother::smoothMeshCuda(
        Mesh& mesh,
        const MeshCrew& crew)
{
    _installCudaSmoother();

    mesh.updateCudaTopology();
    mesh.updateCudaVertices();

    crew.setPluginCudaUniforms(mesh);

    size_t groupCount = mesh.nodeGroups().count();
    uint threadCount = thread::hardware_concurrency();
    mesh.nodeGroups().setCpuWorkgroupSize(threadCount);
    mesh.nodeGroups().setGpuWorkgroupSize(WORKGROUP_SIZE);

    bool isTopoEnabled = crew.topologist()
            .needTopologicalModifications(mesh);

    _smoothPassId = INITIAL_PASS_ID;
    while(evaluateMeshQualityCuda(mesh, crew))
    {
        if(isTopoEnabled)
        {
            verboseCuda = false;
            mesh.fetchCudaVertices();
            crew.topologist().restructureMesh(mesh, crew);
            mesh.updateCudaTopology();
            mesh.updateCudaVertices();
            verboseCuda = true;

            groupCount = mesh.nodeGroups().count();
        }        

        while(evaluateMeshQualityCuda(mesh, crew))
        {
            std::mutex mutex;
            std::condition_variable groupCv;
            std::condition_variable memcpyCv;
            std::atomic<int> moveDone( 0 );
            std::atomic<int> stepDone( 0 );
            std::atomic<bool> memcpyDone( false );

            vector<thread> workers;
            for(uint t=0; t < threadCount; ++t)
            {
                workers.push_back(thread([&, t]() {
                    for(size_t g=0; g < groupCount; ++g)
                    {
                        const NodeGroups::ParallelGroup& group =
                            mesh.nodeGroups().parallelGroups()[g];

                        smoothVertices(mesh, crew,
                            group.cpuOnlyDispatchedNodes[t]);

                        if(g < groupCount)
                        {
                            std::unique_lock<std::mutex> lk(mutex);
                            if(moveDone.fetch_add( 1 ) == threadCount)
                            {
                                ++stepDone;
                                moveDone.store( 0 );
                                memcpyDone = false;
                                groupCv.notify_all();
                            }
                            else
                            {
                                groupCv.wait(lk, [&](){ return stepDone > g; });
                            }

                            memcpyCv.wait(lk, [&](){ return memcpyDone.load(); });
                        }
                    }
                }));
            }


            for(size_t g=0; g < groupCount; ++g)
            {
                const NodeGroups::ParallelGroup& group =
                        mesh.nodeGroups().parallelGroups()[g];

                const NodeGroups::GpuDispatch& dispatch = group.gpuDispatch;
                smoothCudaVertices(dispatch, WORKGROUP_SIZE, _moveCoeff);


                // Fetch subsurface vertex positions from GPU
                fetchCudaSubsurfaceVertices(mesh.verts, group);

                // Synchronize with CPU workers
                if(g < groupCount)
                {
                    std::unique_lock<std::mutex> lk(mutex);
                    if(moveDone.fetch_add( 1 ) == threadCount)
                    {
                        ++stepDone;
                        moveDone.store( 0 );
                        memcpyDone = false;
                        groupCv.notify_all();
                    }
                    else
                    {
                        groupCv.wait(lk, [&](){ return stepDone > g; });
                    }
                }

                // Send boundary vertex positions to GPU
                sendCudaBoundaryVertices(mesh.verts, group);


                // Wake-up threads
                mutex.lock();
                memcpyDone = true;
                mutex.unlock();
                memcpyCv.notify_all();
            }


            for(uint t=0; t < threadCount; ++t)
            {
                workers[t].join();
            }
        }

        if(isTopoEnabled)
            _smoothPassId = COMPARE_PASS_ID;
        else
            break;
    }


    // Fetch new vertices' position
    mesh.fetchCudaVertices();
    mesh.clearCudaMemory();
}

void AbstractVertexWiseSmoother::initializeProgram(
        Mesh& mesh,
        const MeshCrew& crew)
{
    if(_initialized &&
       _samplingShader == crew.sampler().samplingShader() &&
       _evaluationShader == crew.evaluator().evaluationShader() &&
       _measureShader == crew.measurer().measureShader())
            return;


    getLog().postMessage(new Message('I', false,
        "Initializing smoothing compute shader", "AbstractVertexWiseSmoother"));

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
