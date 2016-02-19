#include "AbstractElementWiseSmoother.h"

#include <thread>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <algorithm>
#include <numeric>

#include "DataStructures/MeshCrew.h"
#include "DataStructures/VertexAccum.h"
#include "Samplers/AbstractSampler.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"

using namespace std;
using namespace cellar;


// CUDA Drivers interface
void vertexAccumCudaInstall(size_t vertCount);
void vertexAccumCudaDeinstall();
void smoothCudaElements(
        size_t workGroupCount,
        size_t workgroupSize);
void updateCudaSmoothedElementsVertices(
        const IndependentDispatch& dispatch,
        size_t workgroupSize);


const size_t AbstractElementWiseSmoother::WORKGROUP_SIZE = 256;


AbstractElementWiseSmoother::AbstractElementWiseSmoother(
        const std::vector<std::string>& smoothShaders,
        const installCudaFct installCuda) :
    AbstractSmoother(installCuda),
    _initialized(false),
    _smoothShaders(smoothShaders),
    _vertexAccums(nullptr),
    _accumSsbo(0)
{

}

AbstractElementWiseSmoother::~AbstractElementWiseSmoother()
{

}


void AbstractElementWiseSmoother::smoothMeshSerial(
        Mesh& mesh,
        const MeshCrew& crew)
{
    // Allocate vertex accumulators
    size_t vertCount = mesh.verts.size();
    _vertexAccums = new IVertexAccum*[vertCount];
    for(size_t i=0; i < vertCount; ++i)
        _vertexAccums[i] = new NotThreadSafeVertexAccum();


    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    std::vector<uint> vIds(vertCount);
    std::iota(std::begin(vIds), std::end(vIds), 0);

    _smoothPassId = 0;
    while(evaluateMeshQualitySerial(mesh, crew))
    {
        smoothTets(mesh, crew, 0, tetCount);
        smoothPris(mesh, crew, 0, priCount);
        smoothHexs(mesh, crew, 0, hexCount);

        updateVertexPositions(mesh, crew, vIds);
    }

    mesh.updateVerticesFromCpu();


    // Deallocate vertex accumulators
    for(size_t i=0; i < vertCount; ++i)
        delete _vertexAccums[i];
    delete[] _vertexAccums;
}

void AbstractElementWiseSmoother::smoothMeshThread(
        Mesh& mesh,
        const MeshCrew& crew)
{
    // Allocate vertex accumulators
    size_t vertCount = mesh.verts.size();
    _vertexAccums = new IVertexAccum*[vertCount];
    for(size_t i=0; i < vertCount; ++i)
        _vertexAccums[i] = new ThreadSafeVertexAccum();


    // TODO : Use a thread pool
    size_t groupCount = mesh.independentGroups.size();
    uint threadCount = thread::hardware_concurrency();


    _smoothPassId = 0;
    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    while(evaluateMeshQualityThread(mesh, crew))
    {
        std::mutex mutex;
        std::condition_variable cv;
        std::atomic<int> done( 0 );
        std::atomic<int> step( 0 );

        // Accumulated vertex positions
        vector<thread> workers;
        for(uint t=0; t < threadCount; ++t)
        {
            workers.push_back(thread([&, t]() {
                // Vertex position accumulation
                if(tetCount > 0)
                {
                    size_t tetfirst = (tetCount * t) / threadCount;
                    size_t tetLast = (tetCount * (t+1)) / threadCount;
                    smoothTets(mesh, crew, tetfirst, tetLast);
                }

                if(priCount > 0)
                {
                    size_t prifirst = (priCount * t) / threadCount;
                    size_t priLast = (priCount * (t+1)) / threadCount;
                    smoothPris(mesh, crew, prifirst, priLast);
                }

                if(hexCount > 0)
                {
                    size_t hexfirst = (hexCount * t) / threadCount;
                    size_t hexLast = (hexCount * (t+1)) / threadCount;
                    smoothHexs(mesh, crew, hexfirst, hexLast);
                }

                for(size_t g=0; g < groupCount; ++g)
                {
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


                    // Vertex position update step
                    const std::vector<uint>& group =
                            mesh.independentGroups[g];

                    size_t groupSize = group.size();
                    std::vector<uint> vIds(
                        group.begin() + (groupSize * t) / threadCount,
                        group.begin() + (groupSize * (t+1)) / threadCount);

                    updateVertexPositions(mesh, crew, vIds);
                }
            }));
        }

        for(uint t=0; t < threadCount; ++t)
        {
            workers[t].join();
        }
    }

    mesh.updateVerticesFromCpu();


    // Deallocate vertex accumulators
    for(size_t i=0; i < vertCount; ++i)
        delete _vertexAccums[i];
    delete[] _vertexAccums;
}

void AbstractElementWiseSmoother::smoothMeshGlsl(
        Mesh& mesh,
        const MeshCrew& crew)
{
    initializeProgram(mesh, crew);

    // There's no need to upload vertices again, but absurdly
    // this makes subsequent passes much more faster...
    // I guess it's because the driver put buffer back on GPU.
    // It looks like glGetBufferSubData takes it out of the GPU.
    mesh.updateVerticesFromCpu();

    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    size_t maxElem = glm::max(glm::max(tetCount, priCount), hexCount);
    size_t smoothWgCount = glm::ceil(maxElem / double(WORKGROUP_SIZE));

    struct GpuVertexAccum
    {
        GpuVertexAccum() : posAccum(), weightAccum(0.0) {}
        glm::vec3 posAccum;
        GLfloat weightAccum;
    };

    vector<GpuVertexAccum> accumVec(mesh.verts.size());
    size_t accumSize = sizeof(decltype(accumVec.front())) * accumVec.size();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _accumSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, accumSize, accumVec.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    vector<IndependentDispatch> dispatches;
    organizeDispatches(mesh, WORKGROUP_SIZE, dispatches);
    size_t dispatchCount = dispatches.size();

    _elemSmoothProgram.pushProgram();
    setElementProgramUniforms(mesh, _elemSmoothProgram);
    _elemSmoothProgram.popProgram();

    _vertUpdateProgram.pushProgram();
    setVertexProgramUniforms(mesh, _vertUpdateProgram);
    _vertUpdateProgram.popProgram();

    _smoothPassId = 0;
    while(evaluateMeshQualityGlsl(mesh, crew))
    {
        mesh.bindShaderStorageBuffers();
        GLuint vertAccumBinding = mesh.bufferBinding(
            EBufferBinding::VERTEX_ACCUMS_BUFFER_BINDING);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, vertAccumBinding, _accumSsbo);

        _elemSmoothProgram.pushProgram();
        glDispatchCompute(smoothWgCount, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        _elemSmoothProgram.popProgram();

        _vertUpdateProgram.pushProgram();
        for(size_t d=0; d < dispatchCount; ++d)
        {
            const IndependentDispatch& dispatch = dispatches[d];
            _vertUpdateProgram.setInt("GroupBase", dispatch.base);
            _vertUpdateProgram.setInt("GroupSize", dispatch.size);

            glDispatchCompute(dispatch.workgroupCount, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        _vertUpdateProgram.popProgram();
    }

    // Clear Vertex Accum shader storage buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _accumSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    // Fetch new vertices' position
    mesh.updateVerticesFromGlsl();
}

void AbstractElementWiseSmoother::smoothMeshCuda(
        Mesh& mesh,
        const MeshCrew& crew)
{
    initializeProgram(mesh, crew);
    _installCuda();

    // There's no need to upload vertices again, but absurdly
    // this makes subsequent passes much more faster...
    // I guess it's because the driver put buffer back on GPU.
    // It looks like glGetBufferSubData takes it out of the GPU.
    //mesh.updateVerticesFromCpu();

    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    size_t maxElem = glm::max(glm::max(tetCount, priCount), hexCount);
    size_t smoothWgCount = glm::ceil(maxElem / double(WORKGROUP_SIZE));

    vertexAccumCudaInstall(mesh.verts.size());

    vector<IndependentDispatch> dispatches;
    organizeDispatches(mesh, WORKGROUP_SIZE, dispatches);
    size_t dispatchCount = dispatches.size();

    _smoothPassId = 0;
    while(evaluateMeshQualityCuda(mesh, crew))
    {
        smoothCudaElements(smoothWgCount, WORKGROUP_SIZE);

        for(size_t d=0; d < dispatchCount; ++d)
        {
            const IndependentDispatch& dispatch = dispatches[d];
            updateCudaSmoothedElementsVertices(dispatch, WORKGROUP_SIZE);
        }
    }

    // Clear Vertex Accum shader storage buffer
    vertexAccumCudaDeinstall();


    // Fetch new vertices' position
    mesh.updateVerticesFromCuda();
}

void AbstractElementWiseSmoother::initializeProgram(
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
        "Initializing smoothing compute shader", "AbstractElementWiseSmoother"));

    _modelBoundsShader = mesh.modelBoundsShaderName();
    _samplingShader = crew.sampler().samplingShader();
    _evaluationShader = crew.evaluator().evaluationShader();
    _measureShader = crew.measurer().measureShader();


    // Element Smoothing Program
    _elemSmoothProgram.clearShaders();
    crew.installPlugins(mesh, _elemSmoothProgram);
    _elemSmoothProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        smoothingUtilsShader().c_str()});
    _elemSmoothProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/glsl/compute/Smoothing/ElementWise/VertexAccum.glsl"});
    _elemSmoothProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/glsl/compute/Smoothing/ElementWise/SmoothElements.glsl"});
    for(const string& shader : _smoothShaders)
    {
        _elemSmoothProgram.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            shader.c_str()});
    }

    _elemSmoothProgram.link();
    crew.setPluginUniforms(mesh, _elemSmoothProgram);


    // Update Vertex Positions Program
    _vertUpdateProgram.clearShaders();
    crew.installPlugins(mesh, _vertUpdateProgram);
    _vertUpdateProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        smoothingUtilsShader().c_str()});
    _vertUpdateProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/glsl/compute/Smoothing/ElementWise/VertexAccum.glsl"});
    _vertUpdateProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/glsl/compute/Smoothing/ElementWise/UpdateVertices.glsl"});

    _vertUpdateProgram.link();
    crew.setPluginUniforms(mesh, _vertUpdateProgram);


    // Shader storage vertex accum blocks
    glDeleteBuffers(1, &_accumSsbo);
    _accumSsbo = 0;
    glGenBuffers(1, &_accumSsbo);


    _initialized = true;
}

void AbstractElementWiseSmoother::setElementProgramUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program)
{
}

void AbstractElementWiseSmoother::setVertexProgramUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program)
{
}

void AbstractElementWiseSmoother::printSmoothingParameters(
        const Mesh& mesh,
         OptimizationPlot& plot) const
{
    plot.addSmoothingProperty("Category", "Element-Wise");
}

void AbstractElementWiseSmoother::updateVertexPositions(
        Mesh& mesh,
        const MeshCrew& crew,
        const std::vector<uint>& vIds)
{
    vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;

    size_t vIdCount = vIds.size();
    for(int v = 0; v < vIdCount; ++v)
    {
        uint vId = vIds[v];

        if(!isSmoothable(mesh, vId))
            continue;


        glm::dvec3 pos = verts[vId].p;
        glm::dvec3 posPrim = pos;
        if(_vertexAccums[vId]->assignAverage(posPrim))
        {
            const MeshTopo& topo = topos[vId];
            if(topo.isBoundary)
                posPrim = (*topo.snapToBoundary)(posPrim);

            double patchQuality =
                crew.evaluator().patchQuality(
                    mesh, crew.sampler(), crew.measurer(), vId);

            verts[vId].p = posPrim;

            double patchQualityPrime =
                crew.evaluator().patchQuality(
                    mesh, crew.sampler(), crew.measurer(), vId);

            if(patchQualityPrime < patchQuality)
                verts[vId].p = pos;
        }

        _vertexAccums[vId]->reinit();
    }
}
