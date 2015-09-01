#include "VolumeEdgeEvaluator.h"

#include <GLM/gtx/norm.hpp>

using namespace glm;


VolumeEdgeEvaluator::VolumeEdgeEvaluator() :
    AbstractEvaluator(":/shaders/compute/Quality/VolumeEdge.glsl")
{
}

VolumeEdgeEvaluator::~VolumeEdgeEvaluator()
{

}

double VolumeEdgeEvaluator::tetQuality(const dvec3 vp[]) const
{
    double volume = tetVolume(vp);

    double edge2Sum = 0.0;
    edge2Sum += length2(vp[0] - vp[1]);
    edge2Sum += length2(vp[0] - vp[2]);
    edge2Sum += length2(vp[0] - vp[3]);
    edge2Sum += length2(vp[1] - vp[2]);
    edge2Sum += length2(vp[2] - vp[3]);
    edge2Sum += length2(vp[3] - vp[1]);

    return volume / (edge2Sum*sqrt(edge2Sum))
            / 0.048112522432468815548; // Normalization constant
}

double VolumeEdgeEvaluator::priQuality(const dvec3 vp[]) const
{
    double volume = priVolume(vp);

    double edge2Sum = 0.0;
    edge2Sum += length2(vp[0] - vp[1]);
    edge2Sum += length2(vp[0] - vp[2]);
    edge2Sum += length2(vp[1] - vp[3]);
    edge2Sum += length2(vp[2] - vp[3]);
    edge2Sum += length2(vp[0] - vp[4]);
    edge2Sum += length2(vp[1] - vp[5]);
    edge2Sum += length2(vp[2] - vp[4]);
    edge2Sum += length2(vp[3] - vp[5]);
    edge2Sum += length2(vp[4] - vp[5]);

    // See class doc to understand why the result is saturated
    return min(volume / (edge2Sum*sqrt(edge2Sum))
            / 0.096225044864937631095, 1.0); // Normalization constant
}

double VolumeEdgeEvaluator::hexQuality(const dvec3 vp[]) const
{
    double volume = hexVolume(vp);

    double edge2Sum = 0.0;
    edge2Sum += length2(vp[0] - vp[1]);
    edge2Sum += length2(vp[0] - vp[2]);
    edge2Sum += length2(vp[1] - vp[3]);
    edge2Sum += length2(vp[2] - vp[3]);
    edge2Sum += length2(vp[0] - vp[4]);
    edge2Sum += length2(vp[1] - vp[5]);
    edge2Sum += length2(vp[2] - vp[6]);
    edge2Sum += length2(vp[3] - vp[7]);
    edge2Sum += length2(vp[4] - vp[5]);
    edge2Sum += length2(vp[4] - vp[6]);
    edge2Sum += length2(vp[5] - vp[7]);
    edge2Sum += length2(vp[6] - vp[7]);

    // See class doc to understand why the result is saturated
    return min(volume / (edge2Sum*sqrt(edge2Sum))
            / 0.14433756729740643276, 1.0); // Normalization constant
}
