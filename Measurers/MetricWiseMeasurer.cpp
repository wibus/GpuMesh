#include "MetricWiseMeasurer.h"

#include "DataStructures/Mesh.h"

#include "Samplers/AbstractSampler.h"
#include "Evaluators/AbstractEvaluator.h"


const double DIFF_THRESHOLD = 0.01;

void installCudaMetricWiseMeasurer();


MetricWiseMeasurer::MetricWiseMeasurer() :
    AbstractMeasurer(
        "Metric Wise",
        ":/glsl/compute/Measuring/MetricWise.glsl",
        installCudaMetricWiseMeasurer)
{

}

MetricWiseMeasurer::~MetricWiseMeasurer()
{

}

/* Global segment division
double MetricWiseMeasurer::riemannianDistance(
        const AbstractSampler& sampler,
        const glm::dvec3& a,
        const glm::dvec3& b,
        uint& cachedRefTet) const
{
    glm::dvec3 abDiff = b - a;
    glm::dvec3 middle = (a + b) / 2.0;
    Metric origMetric = sampler.metricAt(middle, cachedRefTet);
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
            Metric metric = sampler.metricAt(segBeg + half_ds, cachedRefTet);
            newDist += glm::sqrt(glm::dot(ds, metric * ds));
            segBeg += ds;
        }

        err = glm::abs(newDist - dist);
        dist = newDist;
    }

    return dist;
}

/*/ // Localize segment division
double MetricWiseMeasurer::riemannianDistance(
        const AbstractSampler& sampler,
        const glm::dvec3& a,
        const glm::dvec3& b,
        uint& cachedRefTet) const
{
    int curr = 0;
    int base = 2;

    double len = 0.0;

    glm::dvec3 d = b-a;
    glm::dvec3 bv = d / double(base);

    while(curr < base)
    {
        double p0 = (curr + 0.5) / base;
        double p1 = (curr + 1.5) / base;

        MeshMetric M0 = sampler.metricAt(a + p0*d, cachedRefTet);
        MeshMetric M1 = sampler.metricAt(a + p1*d, cachedRefTet);

        double l0 = glm::sqrt(glm::dot(bv, M0 * bv));
        double l1 = glm::sqrt(glm::dot(bv, M1 * bv));

        double sum = (l0 + l1);
        double diff = glm::abs(l0 - l1) / (sum/2.0);

        if(diff < DIFF_THRESHOLD)
        {
            len += sum;
            curr += 2;

            if((curr & 0b10) == 0)
            {
                base >>= 1;
                curr >>= 1;
            }

            bv = d / double(base);
        }
        else
        {
            base <<= 1;
            curr <<= 1;
            bv /= 2.0;
        }
    }

    return len;
}
// */

glm::dvec3 MetricWiseMeasurer::riemannianSegment(
        const AbstractSampler& sampler,
        const glm::dvec3& a,
        const glm::dvec3& b,
        uint& cachedRefTet) const
{
    return glm::normalize(b - a) *
            riemannianDistance(sampler, a, b, cachedRefTet);
}

double MetricWiseMeasurer::tetVolume(
        const AbstractSampler& sampler,
        const glm::dvec3 vp[],
        const MeshTet& tet) const
{
    double detSum = glm::determinant(glm::dmat3(
        riemannianSegment(sampler, vp[0], vp[3], tet.c[0]),
        riemannianSegment(sampler, vp[1], vp[3], tet.c[0]),
        riemannianSegment(sampler, vp[2], vp[3], tet.c[0])));

    return detSum / 6.0;
}

double MetricWiseMeasurer::priVolume(
        const AbstractSampler& sampler,
        const glm::dvec3 vp[],
        const MeshPri& pri) const
{
    glm::dvec3 e02 = riemannianSegment(sampler, vp[0], vp[2], pri.c[0]);
    glm::dvec3 e12 = riemannianSegment(sampler, vp[1], vp[2], pri.c[1]);
    glm::dvec3 e32 = riemannianSegment(sampler, vp[3], vp[2], pri.c[3]);
    glm::dvec3 e42 = riemannianSegment(sampler, vp[4], vp[2], pri.c[4]);
    glm::dvec3 e52 = riemannianSegment(sampler, vp[5], vp[2], pri.c[5]);

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
        riemannianSegment(sampler, vp[1], vp[0], hex.c[1]),
        riemannianSegment(sampler, vp[4], vp[0], hex.c[4]),
        riemannianSegment(sampler, vp[3], vp[0], hex.c[3])));
    detSum += glm::determinant(glm::dmat3(
        riemannianSegment(sampler, vp[1], vp[2], hex.c[1]),
        riemannianSegment(sampler, vp[3], vp[2], hex.c[3]),
        riemannianSegment(sampler, vp[6], vp[2], hex.c[6])));
    detSum += glm::determinant(glm::dmat3(
        riemannianSegment(sampler, vp[4], vp[5], hex.c[4]),
        riemannianSegment(sampler, vp[1], vp[5], hex.c[1]),
        riemannianSegment(sampler, vp[6], vp[5], hex.c[6])));
    detSum += glm::determinant(glm::dmat3(
        riemannianSegment(sampler, vp[4], vp[7], hex.c[4]),
        riemannianSegment(sampler, vp[6], vp[7], hex.c[6]),
        riemannianSegment(sampler, vp[3], vp[7], hex.c[3])));
    detSum += glm::determinant(glm::dmat3(
        riemannianSegment(sampler, vp[1], vp[4], hex.c[1]),
        riemannianSegment(sampler, vp[6], vp[4], hex.c[6]),
        riemannianSegment(sampler, vp[3], vp[4], hex.c[3])));

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
    uint& cachedRefTet = verts[vId].c;

    glm::dvec3 forceTotal(0.0);
    uint neigVertCount = topo.neighborVerts.size();
    for(uint n=0; n < neigVertCount; ++n)
    {
        const MeshNeigVert& neigVert = topo.neighborVerts[n];
        const glm::dvec3& npos = verts[neigVert.v];

        forceTotal += computeSpringForce(sampler, pos, npos, cachedRefTet);
    }

    glm::dvec3 equilibrium = pos + forceTotal;
    return equilibrium;
}

glm::dvec3 MetricWiseMeasurer::computeSpringForce(
        const AbstractSampler& sampler,
        const glm::dvec3& pi,
        const glm::dvec3& pj,
        uint& cachedRefTet) const
{
    if(pi == pj)
        return glm::dvec3();

    double d = riemannianDistance(sampler, pi, pj, cachedRefTet);
    glm::dvec3 u = (pi - pj) / d;

    double d2 = d * d;
    //double d4 = d2 * d2;

    //double f = (1 - d4) * glm::exp(-d4);
    //double f = (1-d2)*glm::exp(-d2/4.0)/2.0;
    double f = (1-d2)*glm::exp(-glm::abs(d)/(sqrt(2.0)));

    return f * u;
}
