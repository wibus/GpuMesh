#include "GetmeSmoother.h"

GetmeSmoother::GetmeSmoother() :
    AbstractElementWiseSmoother({":/shader/compute/Smoothing/GETMe.glsl"})
{

}

GetmeSmoother::~GetmeSmoother()
{

}

void GetmeSmoother::smoothTets(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{

}

void GetmeSmoother::smoothPris(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{

}

void GetmeSmoother::smoothHexs(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{

}
