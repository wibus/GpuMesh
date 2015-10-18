#include "MetricWiseMeasurer.h"

#include "DataStructures/Mesh.h"

#include "Discretizers/AbstractDiscretizer.h"
#include "Evaluators/AbstractEvaluator.h"


MetricWiseMeasurer::MetricWiseMeasurer() :
    AbstractMeasurer("Metric Wise", ":/shaders/compute/Measuring/MetricWise.glsl")
{

}

MetricWiseMeasurer::~MetricWiseMeasurer()
{

}

double MetricWiseMeasurer::measuredDistance(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3& a,
        const glm::dvec3& b) const
{
    glm::dvec3 abDiff = b - a;
    glm::dvec3 middle = (a + b) / 2.0;
    double dist = glm::sqrt(glm::dot(abDiff, discretizer.metric(middle) * abDiff));

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
            Metric metric = discretizer.metric(segBeg + half_ds);
            newDist += glm::sqrt(glm::dot(ds, metric * ds));
            segBeg += ds;
        }

        err = glm::abs(newDist - dist);
        dist = newDist;
    }

    return dist;
}

double MetricWiseMeasurer::tetVolume(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3 vp[]) const
{
    return 1.0;
}

double MetricWiseMeasurer::priVolume(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3 vp[]) const
{
    return 1.0;
}

double MetricWiseMeasurer::hexVolume(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3 vp[]) const
{
    return 1.0;
}

double MetricWiseMeasurer::computeLocalElementSize(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        size_t vId) const
{
    return 1.0;
}

glm::dvec3 MetricWiseMeasurer::computeVertexEquilibrium(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        size_t vId) const
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

        forceTotal += computeSpringForce(discretizer, pos, npos);
    }

    glm::dvec3 equilibrium = pos + forceTotal;
    return equilibrium;
}

glm::dvec3 MetricWiseMeasurer::computeSpringForce(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3& pi,
        const glm::dvec3& pj) const
{
    if(pi == pj)
        return glm::dvec3();

    double d = measuredDistance(discretizer, pi, pj);
    glm::dvec3 u = (pi - pj) / d;

    double d2 = d * d;
    double d4 = d2 * d2;

    //double f = (1 - d4) * glm::exp(-d4);
    //double f = (1-d2)*glm::exp(-d2/4.0)/2.0;
    double f = (1-d2)*glm::exp(-glm::abs(d)/(sqrt(2.0)));

    return f * u;
}
