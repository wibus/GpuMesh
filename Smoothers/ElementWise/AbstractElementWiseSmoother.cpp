#include "AbstractElementWiseSmoother.h"

#include <thread>

#include "../SmoothingHelper.h"
#include "Evaluators/AbstractEvaluator.h"

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
    _smoothPassId = 0;
    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    while(evaluateMeshQualitySerial(mesh, evaluator))
    {
        smoothTets(mesh, evaluator, 0, tetCount, false);
        smoothPris(mesh, evaluator, 0, priCount, false);
        smoothHexs(mesh, evaluator, 0, hexCount, false);

        updateVertexPositions(mesh, evaluator);
    }

    mesh.updateGpuVertices();
}

void AbstractElementWiseSmoother::smoothMeshThread(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    // TODO : Use a thread pool

    _smoothPassId = 0;
    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    while(evaluateMeshQualityThread(mesh, evaluator))
    {
        vector<thread> workers;
        uint coreCountHint = thread::hardware_concurrency();
        for(uint t=0; t < coreCountHint; ++t)
        {
            workers.push_back(thread([&, t]() {
                size_t tetfirst = (tetCount * t) / coreCountHint;
                size_t tetLast = (tetCount * (t+1)) / coreCountHint;
                smoothTets(mesh, evaluator, tetfirst, tetLast, true);

                size_t prifirst = (priCount * t) / coreCountHint;
                size_t priLast = (priCount * (t+1)) / coreCountHint;
                smoothPris(mesh, evaluator, prifirst, priLast, true);

                size_t hexfirst = (hexCount * t) / coreCountHint;
                size_t hexLast = (hexCount * (t+1)) / coreCountHint;
                smoothHexs(mesh, evaluator, hexfirst, hexLast, true);
            }));
        }

        for(uint t=0; t < coreCountHint; ++t)
        {
            workers[t].join();
        }

        updateVertexPositions(mesh, evaluator);
    }

    mesh.updateGpuVertices();
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
