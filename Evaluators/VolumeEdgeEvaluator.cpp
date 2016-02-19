#include "VolumeEdgeEvaluator.h"

#include <GLM/gtx/norm.hpp>

#include "Measurers/AbstractMeasurer.h"

using namespace glm;
using namespace cellar;


// CUDA Drivers Interface
void installCudaVolumeEdgeEvaluator()
{
    getLog().postMessage(new Message('W', false,
        "No CUDA implementation of VolumeEdge evaluator",
        "VolumeEdgeEvaluator"));
}

VolumeEdgeEvaluator::VolumeEdgeEvaluator() :
    AbstractEvaluator(":/glsl/compute/Evaluating/VolumeEdge.glsl", installCudaVolumeEdgeEvaluator)
{
}

VolumeEdgeEvaluator::~VolumeEdgeEvaluator()
{

}

double VolumeEdgeEvaluator::tetQuality(
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    double volume = measurer.tetVolume(sampler, vp);

    double edge2Sum = 0.0;
    edge2Sum += length2(vp[0] - vp[1]);
    edge2Sum += length2(vp[0] - vp[2]);
    edge2Sum += length2(vp[0] - vp[3]);
    edge2Sum += length2(vp[1] - vp[2]);
    edge2Sum += length2(vp[2] - vp[3]);
    edge2Sum += length2(vp[3] - vp[1]);

    return volume / (edge2Sum*sqrt(edge2Sum))
            / 0.0080187537387448014348; // Normalization constant
}

double VolumeEdgeEvaluator::priQuality(
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    double volume = measurer.priVolume(sampler, vp);

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
            / 0.016037507477489606339, 1.0); // Normalization constant
}

double VolumeEdgeEvaluator::hexQuality(
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    double volume = measurer.hexVolume(sampler, vp);

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
            / 0.024056261216234407774, 1.0); // Normalization constant
}
