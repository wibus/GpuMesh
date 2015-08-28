#include "AbstractElementWiseSmoother.h"

#include <thread>
#include <atomic>
#include <condition_variable>

#include "../SmoothingHelper.h"
#include "Evaluators/AbstractEvaluator.h"
#include "DataStructures/VertexAccum.h"

using namespace std;
using namespace cellar;


AbstractElementWiseSmoother::AbstractElementWiseSmoother(
        const std::vector<std::string>& smoothShaders)
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

}

void AbstractElementWiseSmoother::initializeProgram(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{

}

void AbstractElementWiseSmoother::updateVertexPositions(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last)
{
    vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;

    for(size_t v = first; v < last; ++v)
    {
        glm::dvec3 pos = verts[v].p;
        glm::dvec3 posPrim = pos;
        if(_vertexAccums[v]->assignAverage(posPrim))
        {
            const MeshTopo& topo = topos[v];
            if(topo.isBoundary)
                posPrim = (*topo.snapToBoundary)(posPrim);

            double patchQuality =
                SmoothingHelper::computePatchQuality(
                    mesh, evaluator, v);

            verts[v].p = posPrim;

            double patchQualityPrime =
                SmoothingHelper::computePatchQuality(
                    mesh, evaluator, v);

            if(patchQualityPrime < patchQuality)
                verts[v].p = pos;
        }
    }
}
