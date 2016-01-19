#include "MetricFreeMeasurer.h"

#include "DataStructures/Mesh.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


MetricFreeMeasurer::MetricFreeMeasurer() :
    AbstractMeasurer("Metric Free", ":/glsl/compute/Measuring/MetricFree.glsl")
{

}

MetricFreeMeasurer::~MetricFreeMeasurer()
{

}

double MetricFreeMeasurer::riemannianDistance(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3& a,
        const glm::dvec3& b) const
{
    return glm::distance(a, b);
}

glm::dvec3 MetricFreeMeasurer::riemannianSegment(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3& a,
        const glm::dvec3& b) const
{
    return b - a;
}

double MetricFreeMeasurer::tetVolume(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3 vp[]) const
{
    double detSum = glm::determinant(glm::dmat3(
        vp[0] - vp[3],
        vp[1] - vp[3],
        vp[2] - vp[3]));

    return detSum / 6.0;
}

double MetricFreeMeasurer::priVolume(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3 vp[]) const
{
    glm::dvec3 e20 = vp[0] - vp[2];
    glm::dvec3 e21 = vp[1] - vp[2];
    glm::dvec3 e23 = vp[3] - vp[2];
    glm::dvec3 e24 = vp[4] - vp[2];
    glm::dvec3 e25 = vp[5] - vp[2];

    double detSum = 0.0;
    detSum += glm::determinant(glm::dmat3(
        e24,
        e20,
        e21));
    detSum += glm::determinant(glm::dmat3(
        e25,
        e21,
        e23));
    detSum += glm::determinant(glm::dmat3(
        e24,
        e21,
        e25));

    return detSum / 6.0;
}

double MetricFreeMeasurer::hexVolume(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3 vp[]) const
{
    double detSum = 0.0;
    detSum += glm::determinant(glm::dmat3(
        vp[1] - vp[0],
        vp[2] - vp[0],
        vp[4] - vp[0]));
    detSum += glm::determinant(glm::dmat3(
        vp[1] - vp[3],
        vp[7] - vp[3],
        vp[2] - vp[3]));
    detSum += glm::determinant(glm::dmat3(
        vp[1] - vp[5],
        vp[4] - vp[5],
        vp[7] - vp[5]));
    detSum += glm::determinant(glm::dmat3(
        vp[2] - vp[6],
        vp[7] - vp[6],
        vp[4] - vp[6]));
    detSum += glm::determinant(glm::dmat3(
        vp[2] - vp[1],
        vp[4] - vp[1],
        vp[7] - vp[1]));

    return detSum / 6.0;
}

glm::dvec3 MetricFreeMeasurer::computeVertexEquilibrium(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        size_t vId) const
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
