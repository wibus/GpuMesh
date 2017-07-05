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
        const NodeGroups::GpuDispatch& dispatch);
void fetchCudaSubsurfaceVertices(
        std::vector<MeshVert>& verts,
        const NodeGroups::ParallelGroup& group);
void sendCudaBoundaryVertices(
        const std::vector<MeshVert>& verts,
        const NodeGroups::ParallelGroup& group);


AbstractVertexWiseSmoother::AbstractVertexWiseSmoother(
        const std::vector<std::string>& smoothShaders,
        const installCudaFct &installCuda,
        const launchCudaKernelFct& launchCudaKernel) :
    _initialized(false),
    _smoothShaders(smoothShaders),
    _installCudaSmoother(installCuda),
    _launchCudaKernel(launchCudaKernel)
{

}

AbstractVertexWiseSmoother::~AbstractVertexWiseSmoother()
{

}

void AbstractVertexWiseSmoother::smoothMeshSerial(
        Mesh& mesh,
        const MeshCrew& crew)
{
    bool isTopoEnabled =
        _schedule.topoOperationEnabled &&
        crew.topologist().needTopologicalModifications(mesh);

    _relocPassId = INITIAL_PASS_ID;
    while(evaluateMeshQualitySerial(mesh, crew))
    {
        if(isTopoEnabled)
        {
            verboseCuda = false;
            crew.topologist().restructureMesh(mesh, crew, _schedule);
            verboseCuda = true;
        }

        while(evaluateMeshQualitySerial(mesh, crew))
        {
            smoothVertices(mesh, crew,
                mesh.nodeGroups().serialGroup());
        }

        if(isTopoEnabled)
            _relocPassId = COMPARE_PASS_ID;
        else
            break;
    }
}

void AbstractVertexWiseSmoother::smoothMeshThread(
        Mesh& mesh,
        const MeshCrew& crew)
{
    bool isTopoEnabled =
        _schedule.topoOperationEnabled &&
        crew.topologist().needTopologicalModifications(mesh);

    // TODO : Use a thread pool
    uint threadCount = thread::hardware_concurrency();    
    mesh.nodeGroups().setCpuWorkerCount(threadCount);

    _relocPassId = INITIAL_PASS_ID;
    while(evaluateMeshQualityThread(mesh, crew))
    {
        if(isTopoEnabled)
        {
            verboseCuda = false;
            crew.topologist().restructureMesh(mesh, crew, _schedule);
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
            _relocPassId = COMPARE_PASS_ID;
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
    crew.updateGlslData(mesh);

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
    mesh.nodeGroups().setCpuWorkerCount(threadCount);
    mesh.nodeGroups().setGpuDispatcher(glslDispatcher());

    bool isTopoEnabled =
        _schedule.topoOperationEnabled &&
        crew.topologist().needTopologicalModifications(mesh);

    _relocPassId = INITIAL_PASS_ID;
    while(evaluateMeshQualityGlsl(mesh, crew))
    {
        if(isTopoEnabled)
        {
            verboseCuda = false;
            mesh.fetchGlslVertices();
            crew.topologist().restructureMesh(mesh, crew, _schedule);
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


                glBindBuffer(GL_SHADER_STORAGE_BUFFER,
                             mesh.glBuffer(EMeshBuffer::VERT));

                if(dispatch.workgroupCount.x *
                   dispatch.workgroupCount.y *
                   dispatch.workgroupCount.z > 0)
                {
                    _vertSmoothProgram.setInt("GroupBase", dispatch.gpuBufferBase);
                    _vertSmoothProgram.setInt("GroupSize", dispatch.gpuBufferSize);

                    glDispatchComputeGroupSizeARB(
                        dispatch.workgroupCount.x,
                        dispatch.workgroupCount.y,
                        dispatch.workgroupCount.z,
                        dispatch.workgroupSize.x,
                        dispatch.workgroupSize.y,
                        dispatch.workgroupSize.z);

                    glMemoryBarrier(GL_ALL_BARRIER_BITS);


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
            _relocPassId = COMPARE_PASS_ID;
        else
            break;
    }


    // Fetch new vertex positions
    mesh.fetchGlslVertices();
    mesh.clearGlslMemory();
    crew.clearGlslMemory(mesh);
}

void AbstractVertexWiseSmoother::smoothMeshCuda(
        Mesh& mesh,
        const MeshCrew& crew)
{
    _installCudaSmoother();

    mesh.updateCudaTopology();
    mesh.updateCudaVertices();
    crew.updateCudaData(mesh);

    crew.setPluginCudaUniforms(mesh);

    size_t groupCount = mesh.nodeGroups().count();
    uint threadCount = thread::hardware_concurrency();
    mesh.nodeGroups().setCpuWorkerCount(threadCount);
    mesh.nodeGroups().setGpuDispatcher(cudaDispatcher());

    bool isTopoEnabled =
        _schedule.topoOperationEnabled &&
        crew.topologist().needTopologicalModifications(mesh);

    _relocPassId = INITIAL_PASS_ID;
    while(evaluateMeshQualityCuda(mesh, crew))
    {
        if(isTopoEnabled)
        {
            verboseCuda = false;
            mesh.fetchCudaVertices();
            crew.topologist().restructureMesh(mesh, crew, _schedule);
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

                if(dispatch.workgroupCount.x *
                   dispatch.workgroupCount.y *
                   dispatch.workgroupCount.z > 0)
                {
                    _launchCudaKernel(dispatch);

                    // Fetch subsurface vertex positions from GPU
                    fetchCudaSubsurfaceVertices(mesh.verts, group);
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
            _relocPassId = COMPARE_PASS_ID;
        else
            break;
    }


    // Fetch new vertex positions
    mesh.fetchCudaVertices();
    mesh.clearCudaMemory();
    crew.clearCudaMemory(mesh);
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

    std::string launcher = glslLauncher();

    _vertSmoothProgram.reset();
    crew.installPlugins(mesh, _vertSmoothProgram);
    _vertSmoothProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        smoothingUtilsShader().c_str()}
    );
    if(!launcher.empty())
    {
        _vertSmoothProgram.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            launcher});
    }
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
}

void AbstractVertexWiseSmoother::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationImpl& plotImpl) const
{
    plotImpl.addSmoothingProperty("Category", "Vertex-Wise");
}

std::string AbstractVertexWiseSmoother::glslLauncher() const
{
    return ":/glsl/compute/Smoothing/VertexWise/SmoothVertices.glsl";
}

NodeGroups::GpuDispatcher AbstractVertexWiseSmoother::glslDispatcher() const
{
    return [this](NodeGroups::GpuDispatch& d)
    {
        d.workgroupSize = glm::uvec3(_glslThreadCount, 1, 1);
        d.workgroupCount = glm::uvec3(
            glm::ceil(double(d.gpuBufferSize)/_glslThreadCount), 1, 1);
    };
}

NodeGroups::GpuDispatcher AbstractVertexWiseSmoother::cudaDispatcher() const
{
    return [this](NodeGroups::GpuDispatch& d)
    {
        d.workgroupSize = glm::uvec3(_cudaThreadCount, 1, 1);
        d.workgroupCount = glm::uvec3(
            glm::ceil(double(d.gpuBufferSize)/_cudaThreadCount), 1, 1);
    };
}
