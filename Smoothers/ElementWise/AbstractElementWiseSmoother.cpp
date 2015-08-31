#include "AbstractElementWiseSmoother.h"

#include <iostream>
#include <thread>
#include <atomic>
#include <condition_variable>

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


    _smoothPassId = 0;
    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    while(evaluateMeshQualitySerial(mesh, evaluator))
    {
        smoothTets(mesh, evaluator, 0, tetCount);
        smoothPris(mesh, evaluator, 0, priCount);
        smoothHexs(mesh, evaluator, 0, hexCount);

        updateVertexPositions(mesh, evaluator, 0, vertCount);
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


    _smoothPassId = 0;
    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    while(evaluateMeshQualityThread(mesh, evaluator))
    {
        uint coreCountHint = thread::hardware_concurrency();

        // TODO : Use a thread pool
        std::mutex doneMutex;
        std::mutex stepMutex;
        std::condition_variable doneCv;
        std::condition_variable stepCv;
        std::vector<bool> threadDone(coreCountHint, false);
        bool nextStep = false;

        // Accumulated vertex positions
        vector<thread> workers;
        for(uint t=0; t < coreCountHint; ++t)
        {
            workers.push_back(thread([&, t]() {
                // Vertex position accumulation
                if(tetCount > 0)
                {
                    size_t tetfirst = (tetCount * t) / coreCountHint;
                    size_t tetLast = (tetCount * (t+1)) / coreCountHint;
                    smoothTets(mesh, evaluator, tetfirst, tetLast);
                }

                if(priCount > 0)
                {
                    size_t prifirst = (priCount * t) / coreCountHint;
                    size_t priLast = (priCount * (t+1)) / coreCountHint;
                    smoothPris(mesh, evaluator, prifirst, priLast);
                }

                if(hexCount > 0)
                {
                    size_t hexfirst = (hexCount * t) / coreCountHint;
                    size_t hexLast = (hexCount * (t+1)) / coreCountHint;
                    smoothHexs(mesh, evaluator, hexfirst, hexLast);
                }

                // Now that vertex new positions were accumulated
                // We wait for every worker to terminate in order
                // to start the vertex update step.
                {
                    std::lock_guard<std::mutex> lk(doneMutex);
                    threadDone[t] = true;
                }
                doneCv.notify_one();

                {
                    std::unique_lock<std::mutex> lk(stepMutex);
                    stepCv.wait(lk, [&](){ return nextStep; });
                }

                // Vertex position update step
                size_t vertFirst = (vertCount * t) / coreCountHint;
                size_t vertLast = (vertCount * (t+1)) / coreCountHint;
                updateVertexPositions(mesh, evaluator, vertFirst, vertLast);
            }));
        }

        // Wait for thread to finish vertex position accumulation
        {
            std::unique_lock<std::mutex> lk(doneMutex);
            doneCv.wait(lk, [&](){
                bool allFinished = true;
                for(uint t=0; t < coreCountHint; ++t)
                    allFinished = allFinished && threadDone[t];
                return allFinished;
            });
        }

        // Notify threads to begin vertex position update
        {
            std::lock_guard<std::mutex> lk(stepMutex);
            nextStep = true;
        }
        stepCv.notify_all();

        for(uint t=0; t < coreCountHint; ++t)
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
        glDispatchCompute(updateWgCount, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        _updateProgram.popProgram();

        /*
        if(_smoothPassId == 1)
        {
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, _accumSsbo);
            GpuVertexAccum* data = (GpuVertexAccum*) glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

            for(size_t vId=0; vId < vertCount; ++vId)
            {
                std::cout <<
                    "[(" << data[vId].posAccum.x << ", " <<
                            data[vId].posAccum.y << ", " <<
                            data[vId].posAccum.z  << "), " <<
                        data[vId].weightAccum << "] \t";
            }
            std::cout << std::endl;

            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
        */
    }


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
        size_t first,
        size_t last)
{
    vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;

    for(size_t vId = first; vId < last; ++vId)
    {
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
