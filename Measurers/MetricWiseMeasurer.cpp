#include "MetricWiseMeasurer.h"

#include "DataStructures/Mesh.h"

#include "Samplers/AbstractSampler.h"
#include "Evaluators/AbstractEvaluator.h"


void installCudaMetricWiseMeasurer();


MetricWiseMeasurer::MetricWiseMeasurer() :
    AbstractMeasurer("Metric Wise", ":/glsl/compute/Measuring/MetricWise.glsl", installCudaMetricWiseMeasurer)
{

}

MetricWiseMeasurer::~MetricWiseMeasurer()
{

}

double MetricWiseMeasurer::riemannianDistance(
        const AbstractSampler& sampler,
        const glm::dvec3& a,
        const glm::dvec3& b,
        uint vId) const
{
    glm::dvec3 abDiff = b - a;
    glm::dvec3 middle = (a + b) / 2.0;
    Metric origMetric = sampler.metricAt(middle, vId);
    double dist = glm::sqrt(glm::dot(abDiff, origMetric * abDiff));

    int segmentCount = 1;
    double err = 1.0;
    while(err > 1e-5)
    {
        segmentCount *= 2;
        glm::dvec3 segBeg = a;
        glm::dvec3 ds = abDiff / double(segmentCount);
        glm::dvec3 half_ds = ds / 2.0;

        double newDist = 0.0;
        for(int i=0; i < segmentCount; ++i)
        {
            Metric metric = sampler.metricAt(segBeg + half_ds, vId);
            newDist += glm::sqrt(glm::dot(ds, metric * ds));
            segBeg += ds;
        }

        err = glm::abs(newDist - dist);
        dist = newDist;
    }

    return dist;
}

glm::dvec3 MetricWiseMeasurer::riemannianSegment(
        const AbstractSampler& sampler,
        const glm::dvec3& a,
        const glm::dvec3& b,
        uint vId) const
{
    return glm::normalize(b - a) *
            riemannianDistance(sampler, a, b, vId);
}

double MetricWiseMeasurer::tetVolume(
        const AbstractSampler& sampler,
        const glm::dvec3 vp[],
        const MeshTet& tet) const
{
    double detSum = glm::determinant(glm::dmat3(
        riemannianSegment(sampler, vp[0], vp[3], tet.v[0]),
        riemannianSegment(sampler, vp[1], vp[3], tet.v[1]),
        riemannianSegment(sampler, vp[2], vp[3], tet.v[2])));

    return detSum / 6.0;
}

double MetricWiseMeasurer::priVolume(
        const AbstractSampler& sampler,
        const glm::dvec3 vp[],
        const MeshPri& pri) const
{
    glm::dvec3 e02 = riemannianSegment(sampler, vp[0], vp[2], pri.v[0]);
    glm::dvec3 e12 = riemannianSegment(sampler, vp[1], vp[2], pri.v[1]);
    glm::dvec3 e32 = riemannianSegment(sampler, vp[3], vp[2], pri.v[3]);
    glm::dvec3 e42 = riemannianSegment(sampler, vp[4], vp[2], pri.v[4]);
    glm::dvec3 e52 = riemannianSegment(sampler, vp[5], vp[2], pri.v[5]);

    double detSum = 0.0;
    detSum += glm::determinant(glm::dmat3(e32, e52, e42));
    detSum += glm::determinant(glm::dmat3(e02, e32, e42));
    detSum += glm::determinant(glm::dmat3(e12, e02, e42));

    return detSum / 6.0;
}

double MetricWiseMeasurer::hexVolume(
        const AbstractSampler& sampler,
        const glm::dvec3 vp[],
        const MeshHex& hex) const
{
    double detSum = 0.0;
    detSum += glm::determinant(glm::dmat3(
        riemannianSegment(sampler, vp[1], vp[0], hex.v[1]),
        riemannianSegment(sampler, vp[4], vp[0], hex.v[4]),
        riemannianSegment(sampler, vp[3], vp[0], hex.v[3])));
    detSum += glm::determinant(glm::dmat3(
        riemannianSegment(sampler, vp[1], vp[2], hex.v[1]),
        riemannianSegment(sampler, vp[3], vp[2], hex.v[3]),
        riemannianSegment(sampler, vp[6], vp[2], hex.v[6])));
    detSum += glm::determinant(glm::dmat3(
        riemannianSegment(sampler, vp[4], vp[5], hex.v[4]),
        riemannianSegment(sampler, vp[1], vp[5], hex.v[1]),
        riemannianSegment(sampler, vp[6], vp[5], hex.v[6])));
    detSum += glm::determinant(glm::dmat3(
        riemannianSegment(sampler, vp[4], vp[7], hex.v[4]),
        riemannianSegment(sampler, vp[6], vp[7], hex.v[6]),
        riemannianSegment(sampler, vp[3], vp[7], hex.v[3])));
    detSum += glm::determinant(glm::dmat3(
        riemannianSegment(sampler, vp[1], vp[4], hex.v[1]),
        riemannianSegment(sampler, vp[6], vp[4], hex.v[6]),
        riemannianSegment(sampler, vp[3], vp[4], hex.v[3])));

    return detSum / 6.0;
}

glm::dvec3 MetricWiseMeasurer::computeVertexEquilibrium(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        uint vId) const
{
    const std::vector<MeshVert>& verts = mesh.verts;

    const MeshTopo& topo = mesh.topos[vId];
    const glm::dvec3& pos = verts[vId].p;

    glm::dvec3 forceTotal(0.0);
    uint neigVertCount = topo.neighborVerts.size();
    for(uint n=0; n < neigVertCount; ++n)
    {
        const MeshNeigVert& neigVert = topo.neighborVerts[n];
        const glm::dvec3& npos = verts[neigVert.v];

        forceTotal += computeSpringForce(sampler, pos, npos, vId);
    }

    glm::dvec3 equilibrium = pos + forceTotal;
    return equilibrium;
}

glm::dvec3 MetricWiseMeasurer::computeSpringForce(
        const AbstractSampler& sampler,
        const glm::dvec3& pi,
        const glm::dvec3& pj,
        uint vId) const
{
    if(pi == pj)
        return glm::dvec3();

    double d = riemannianDistance(sampler, pi, pj, vId);
    glm::dvec3 u = (pi - pj) / d;

    double d2 = d * d;
    //double d4 = d2 * d2;

    //double f = (1 - d4) * glm::exp(-d4);
    //double f = (1-d2)*glm::exp(-d2/4.0)/2.0;
    double f = (1-d2)*glm::exp(-glm::abs(d)/(sqrt(2.0)));

    return f * u;
}
