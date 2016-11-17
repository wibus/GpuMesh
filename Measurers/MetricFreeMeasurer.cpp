#include "MetricFreeMeasurer.h"

#include "DataStructures/Mesh.h"
#include "Samplers/AbstractSampler.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


void installCudaMetricFreeMeasurer();


MetricFreeMeasurer::MetricFreeMeasurer() :
    AbstractMeasurer(
        "Metric Free",
        ":/glsl/compute/Measuring/MetricFree.glsl",
        installCudaMetricFreeMeasurer)
{

}

MetricFreeMeasurer::~MetricFreeMeasurer()
{

}

double MetricFreeMeasurer::riemannianDistance(
        const AbstractSampler& sampler,
        const glm::dvec3& a,
        const glm::dvec3& b,
        uint& cachedRefTet) const
{
    return glm::distance(a, b) * sampler.scaling();
}

glm::dvec3 MetricFreeMeasurer::riemannianSegment(
        const AbstractSampler& sampler,
        const glm::dvec3& a,
        const glm::dvec3& b,
        uint& cachedRefTet) const
{
    return (b - a) * sampler.scaling();
}

double MetricFreeMeasurer::tetVolume(
        const AbstractSampler& sampler,
        const glm::dvec3 vp[],
        const MeshTet& tet) const
{
    double detSum = glm::determinant(glm::dmat3(
        vp[3] - vp[0],
        vp[3] - vp[1],
        vp[3] - vp[2]));

    return sampler.scalingCube() * detSum / 6.0;
}

double MetricFreeMeasurer::priVolume(
        const AbstractSampler& sampler,
        const glm::dvec3 vp[],
        const MeshPri& pri) const
{
    glm::dvec3 e02 = vp[2] - vp[0];
    glm::dvec3 e12 = vp[2] - vp[1];
    glm::dvec3 e32 = vp[2] - vp[3];
    glm::dvec3 e42 = vp[2] - vp[4];
    glm::dvec3 e52 = vp[2] - vp[5];

    double detSum = 0.0;
    detSum += glm::determinant(glm::dmat3(e32, e52, e42));
    detSum += glm::determinant(glm::dmat3(e02, e32, e42));
    detSum += glm::determinant(glm::dmat3(e12, e02, e42));

    return sampler.scalingCube() * detSum / 6.0;
}

double MetricFreeMeasurer::hexVolume(
        const AbstractSampler& sampler,
        const glm::dvec3 vp[],
        const MeshHex& hex) const
{
    double detSum = 0.0;
    detSum += glm::determinant(glm::dmat3(
        vp[0] - vp[1],
        vp[0] - vp[4],
        vp[0] - vp[3]));
    detSum += glm::determinant(glm::dmat3(
        vp[2] - vp[1],
        vp[2] - vp[3],
        vp[2] - vp[6]));
    detSum += glm::determinant(glm::dmat3(
        vp[5] - vp[4],
        vp[5] - vp[1],
        vp[5] - vp[6]));
    detSum += glm::determinant(glm::dmat3(
        vp[7] - vp[4],
        vp[7] - vp[6],
        vp[7] - vp[3]));
    detSum += glm::determinant(glm::dmat3(
        vp[4] - vp[1],
        vp[4] - vp[6],
        vp[4] - vp[3]));

    return sampler.scalingCube() * detSum / 6.0;
}

glm::dvec3 MetricFreeMeasurer::computeVertexEquilibrium(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        uint vId) const
{
    const std::vector<MeshVert>& verts = mesh.verts;
    const std::vector<MeshTet>& tets = mesh.tets;
    const std::vector<MeshPri>& pris = mesh.pris;
    const std::vector<MeshHex>& hexs = mesh.hexs;

    const MeshTopo& topo = mesh.topos[vId];

    uint totalVertCount = 0;
    glm::dvec3 patchCenter(0.0);
    uint neigElemCount = topo.neighborElems.size();
    for(uint n=0; n < neigElemCount; ++n)
    {
        const MeshNeigElem& neigElem = topo.neighborElems[n];
        switch(neigElem.type)
        {
        case MeshTet::ELEMENT_TYPE:
            totalVertCount += MeshTet::VERTEX_COUNT - 1;
            for(uint i=0; i < MeshTet::VERTEX_COUNT; ++i)
                patchCenter += verts[tets[neigElem.id].v[i]].p;
            break;

        case MeshPri::ELEMENT_TYPE:
            totalVertCount += MeshPri::VERTEX_COUNT - 1;
            for(uint i=0; i < MeshPri::VERTEX_COUNT; ++i)
                patchCenter += verts[pris[neigElem.id].v[i]].p;
            break;

        case MeshHex::ELEMENT_TYPE:
            totalVertCount += MeshHex::VERTEX_COUNT - 1;
            for(uint i=0; i < MeshHex::VERTEX_COUNT; ++i)
                patchCenter += verts[hexs[neigElem.id].v[i]].p;
            break;
        }
    }

    const glm::dvec3& pos = verts[vId].p;
    patchCenter = (patchCenter - pos * double(neigElemCount))
                    / double(totalVertCount);

    return patchCenter;
}
