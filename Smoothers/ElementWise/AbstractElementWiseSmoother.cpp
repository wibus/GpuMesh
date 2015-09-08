#include "AbstractElementWiseSmoother.h"

#include <thread>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <algorithm>

#include "../SmoothingHelper.h"
#include "Evaluators/AbstractEvaluator.h"
#include "DataStructures/VertexAccum.h"

using namespace std;
using namespace cellar;


const size_t AbstractElementWiseSmoother::WORKGROUP_SIZE = 256;


AbstractElementWiseSmoother::AbstractElementWiseSmoother(
        int dispatchMode,
        const std::vector<std::string>& smoothShaders) :
    _initialized(false),
    _dispatchMode(dispatchMode),
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
        AbstractEvaluator& evaluator)
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
    while(evaluateMeshQualitySerial(mesh, evaluator))
    {
        smoothTets(mesh, evaluator, 0, tetCount);
        smoothPris(mesh, evaluator, 0, priCount);
        smoothHexs(mesh, evaluator, 0, hexCount);

        updateVertexPositions(mesh, evaluator, vIds);
    }

    mesh.updateGpuVertices();


    // Deallocate vertex accumulators
    for(size_t i=0; i < vertCount; ++i)
        delete _vertexAccums[i];
    delete[] _vertexAccums;
}

void AbstractElementWiseSmoother::smoothMeshThread(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    // Allocate vertex accumulators
    size_t vertCount = mesh.verts.size();
    _vertexAccums = new IVertexAccum*[vertCount];
    for(size_t i=0; i < vertCount; ++i)
        _vertexAccums[i] = new ThreadSafeVertexAccum();


    // TODO : Use a thread pool
    size_t groupCount = mesh.exclusiveGroups.size();
    uint threadCount = thread::hardware_concurrency();


    _smoothPassId = 0;
    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    while(evaluateMeshQualityThread(mesh, evaluator))
    {
        std::mutex mutex;
        std::condition_variable cv;
        std::atomic_int done( 0 );
        std::atomic_int step( 0 );

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
                    smoothTets(mesh, evaluator, tetfirst, tetLast);
                }

                if(priCount > 0)
                {
                    size_t prifirst = (priCount * t) / threadCount;
                    size_t priLast = (priCount * (t+1)) / threadCount;
                    smoothPris(mesh, evaluator, prifirst, priLast);
                }

                if(hexCount > 0)
                {
                    size_t hexfirst = (hexCount * t) / threadCount;
                    size_t hexLast = (hexCount * (t+1)) / threadCount;
                    smoothHexs(mesh, evaluator, hexfirst, hexLast);
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
                            mesh.exclusiveGroups[g];

                    size_t groupSize = group.size();
                    std::vector<uint> vIds(
                        group.begin() + (groupSize * t) / threadCount,
                        group.begin() + (groupSize * (t+1)) / threadCount);

                    updateVertexPositions(mesh, evaluator, vIds);
                }
            }));
        }

        for(uint t=0; t < threadCount; ++t)
        {
            workers[t].join();
        }
    }

    mesh.updateGpuVertices();


    // Deallocate vertex accumulators
    for(size_t i=0; i < vertCount; ++i)
        delete _vertexAccums[i];
    delete[] _vertexAccums;
}

void AbstractElementWiseSmoother::smoothMeshGlsl(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    initializeProgram(mesh, evaluator);

    // There's no need to upload vertices again, but absurdly
    // this makes subsequent passes much more faster...
    // I guess it's because the driver put buffer back on GPU.
    // It looks like glGetBufferSubData takes it out of the GPU.
    mesh.updateGpuVertices();

    size_t vertCount = mesh.verts.size();
    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    size_t maxElem = glm::max(glm::max(tetCount, priCount), hexCount);
    size_t smoothWgCount = glm::ceil(maxElem / double(WORKGROUP_SIZE));
    size_t updateWgCount = glm::ceil(vertCount / double(WORKGROUP_SIZE));

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


    _smoothingProgram.pushProgram();
    _smoothingProgram.setFloat("Lambda", 0.78);
    _smoothingProgram.setInt("DispatchMode", _dispatchMode);
    _smoothingProgram.popProgram();

    _updateProgram.pushProgram();
    _updateProgram.setFloat("Lambda", 0.78);
    _updateProgram.setInt("DispatchMode", _dispatchMode);
    _updateProgram.popProgram();

    _smoothPassId = 0;
    while(evaluateMeshQualityGlsl(mesh, evaluator))
    {
        mesh.bindShaderStorageBuffers();        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,
                         mesh.firstFreeBufferBinding(), _accumSsbo);

        _smoothingProgram.pushProgram();
        glDispatchCompute(smoothWgCount, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        _smoothingProgram.popProgram();

        _updateProgram.pushProgram();
        if(_dispatchMode == SmoothingHelper::DISPATCH_MODE_EXCLUSIVE)
        {
            size_t base = 0;
            for(const std::vector<uint>& group : mesh.exclusiveGroups)
            {
                size_t size = group.size();
                _updateProgram.setInt("GroupBase", base);
                _updateProgram.setInt("GroupSize", size);
                base += size;

                updateWgCount = glm::ceil(size / double(WORKGROUP_SIZE));
                glDispatchCompute(updateWgCount, 1, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
        }
        else
        {
            glDispatchCompute(updateWgCount, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        _updateProgram.popProgram();
    }

    // Clear Vertex Accum shader storage buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _accumSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    // Fetch new vertices' position
    mesh.updateCpuVertices();
}

void AbstractElementWiseSmoother::initializeProgram(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    if(_initialized &&
       _modelBoundsShader == mesh.modelBoundsShaderName() &&
       _shapeMeasureShader == evaluator.shapeMeasureShader())
        return;


    getLog().postMessage(new Message('I', false,
        "Initializing smoothing compute shader", "AbstractElementWiseSmoother"));

    _modelBoundsShader = mesh.modelBoundsShaderName();
    _shapeMeasureShader = evaluator.shapeMeasureShader();


    // Element Smoothing Program
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
    _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Smoothing/ElementWise/VertexAccum.glsl"});
    _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Smoothing/ElementWise/SmoothElements.glsl"});
    for(const string& shader : _smoothShaders)
    {
        _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            shader.c_str()});
    }
    _smoothingProgram.link();

    mesh.uploadGeometry(_smoothingProgram);


    // Update Vertex Positions Program
    _updateProgram.clearShaders();
    _updateProgram.addShader(GL_COMPUTE_SHADER,
        _modelBoundsShader);
    _updateProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        _shapeMeasureShader.c_str()});
    _updateProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Quality/QualityInterface.glsl"});
    _updateProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        SmoothingHelper::shaderName().c_str()});
    _updateProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Smoothing/ElementWise/VertexAccum.glsl"});
    _updateProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Smoothing/ElementWise/UpdateVertices.glsl"});
    _updateProgram.link();

    mesh.uploadGeometry(_updateProgram);


    // Shader storage vertex accum blocks
    glDeleteBuffers(1, &_accumSsbo);
    _accumSsbo = 0;
    glGenBuffers(1, &_accumSsbo);


    _initialized = true;
}

void AbstractElementWiseSmoother::updateVertexPositions(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        const std::vector<uint>& vIds)
{
    vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;

    size_t vIdCount = vIds.size();
    for(int v = 0; v < vIdCount; ++v)
    {
        uint vId = vIds[v];

        glm::dvec3 pos = verts[vId].p;
        glm::dvec3 posPrim = pos;
        if(_vertexAccums[vId]->assignAverage(posPrim))
        {
            const MeshTopo& topo = topos[vId];
            if(topo.isBoundary)
                posPrim = (*topo.snapToBoundary)(posPrim);

            double patchQuality =
                SmoothingHelper::computePatchQuality(
                    mesh, evaluator, vId);

            verts[vId].p = posPrim;

            double patchQualityPrime =
                SmoothingHelper::computePatchQuality(
                    mesh, evaluator, vId);

            if(patchQualityPrime < patchQuality)
                verts[vId].p = pos;
        }

        _vertexAccums[vId]->reinit();
    }
}
