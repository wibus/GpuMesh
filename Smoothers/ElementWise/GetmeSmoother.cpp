#include "GetmeSmoother.h"

#include <mutex>

#include "../SmoothingHelper.h"
#include "Evaluators/AbstractEvaluator.h"
#include "DataStructures/VertexAccum.h"

using namespace std;


GetmeSmoother::GetmeSmoother() :
    AbstractElementWiseSmoother({":/shader/compute/Smoothing/GETMe.glsl"}),
    _lambda(0.78)
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
    const vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;
    const vector<MeshTet>& tets = mesh.tets;

    for(int e = first; e < last; ++e)
    {
        const MeshTet& tet = tets[e];

        uint vi[] = {
            tet.v[0],
            tet.v[1],
            tet.v[2],
            tet.v[3],
        };

        glm::dvec3 vp[] = {
            verts[vi[0]],
            verts[vi[1]],
            verts[vi[2]],
            verts[vi[3]],
        };

        glm::dvec3 center = 0.25 * (
            vp[0] + vp[1] + vp[2] + vp[3]);

        double volume =
            glm::determinant(
                glm::dmat3(
                    vp[0] - vp[3],
                    vp[1] - vp[3],
                    vp[2] - vp[3]));
        if(volume <= 0.0)
            continue;

        double quality = evaluator.tetQuality(vp);

        glm::dvec3 n[] = {
            glm::cross(vp[3]-vp[1], vp[1]-vp[2]),
            glm::cross(vp[3]-vp[2], vp[2]-vp[0]),
            glm::cross(vp[1]-vp[3], vp[3]-vp[0]),
            glm::cross(vp[1]-vp[0], vp[0]-vp[2]),
        };

        vp[0] = vp[0] + _lambda * n[0] / glm::sqrt(glm::length(n[0]));
        vp[1] = vp[1] + _lambda * n[1] / glm::sqrt(glm::length(n[1]));
        vp[2] = vp[2] + _lambda * n[2] / glm::sqrt(glm::length(n[2]));
        vp[3] = vp[3] + _lambda * n[3] / glm::sqrt(glm::length(n[3]));

        double volumePrime =
            glm::determinant(
                glm::dmat3(
                    vp[0] - vp[3],
                    vp[1] - vp[3],
                    vp[2] - vp[3]));
        if(volumePrime <= 0.0)
            continue;

        double volumeVar = glm::pow(volume / volumePrime, 1.0/3.0);

        vp[0] = center + volumeVar * (vp[0] - center);
        vp[1] = center + volumeVar * (vp[1] - center);
        vp[2] = center + volumeVar * (vp[2] - center);
        vp[3] = center + volumeVar * (vp[3] - center);

        if(topos[vi[0]].isBoundary) vp[0] = (*topos[vi[0]].snapToBoundary)(vp[0]);
        if(topos[vi[1]].isBoundary) vp[1] = (*topos[vi[1]].snapToBoundary)(vp[1]);
        if(topos[vi[2]].isBoundary) vp[2] = (*topos[vi[2]].snapToBoundary)(vp[2]);
        if(topos[vi[3]].isBoundary) vp[3] = (*topos[vi[3]].snapToBoundary)(vp[3]);

        double qualityPrime = evaluator.tetQuality(vp);
        double weight = qualityPrime / quality;

        _vertexAccums[tet[0]]->add(vp[0], weight);
        _vertexAccums[tet[1]]->add(vp[1], weight);
        _vertexAccums[tet[2]]->add(vp[2], weight);
        _vertexAccums[tet[3]]->add(vp[3], weight);
    }
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
