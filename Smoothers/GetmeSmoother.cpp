#include "GetmeSmoother.h"

GetmeSmoother::GetmeSmoother() :
    AbstractSmoother({":/shader/compute/Smoothing/GETMe.glsl"})
{

}

GetmeSmoother::~GetmeSmoother()
{

}

void GetmeSmoother::smoothMeshSerial(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{

}

void GetmeSmoother::smoothMeshThread(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{

}

void GetmeSmoother::smoothMeshGlsl(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{

}

void GetmeSmoother::smoothVertices(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{

}
