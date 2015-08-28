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
        size_t last)
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

        glm::dvec3 n[] = {
            glm::cross(vp[3]-vp[1], vp[1]-vp[2]),
            glm::cross(vp[3]-vp[2], vp[2]-vp[0]),
            glm::cross(vp[1]-vp[3], vp[3]-vp[0]),
            glm::cross(vp[1]-vp[0], vp[0]-vp[2]),
        };

        double volume =
            glm::determinant(
                glm::dmat3(
                    vp[0] - vp[3],
                    vp[1] - vp[3],
                    vp[2] - vp[3]));

        double quality = evaluator.tetQuality(vp);

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

        double absVolumeRation = glm::abs(volume / volumePrime);
        double volumeVar = glm::pow(absVolumeRation, 1.0/3.0);

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
        size_t last)
{

}

void GetmeSmoother::smoothHexs(Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last)
{
    const vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;
    const vector<MeshHex>& hexs = mesh.hexs;

    for(int e = first; e < last; ++e)
    {
        const MeshHex& hex = hexs[e];

        uint vi[] = {
            hex.v[0],
            hex.v[1],
            hex.v[2],
            hex.v[3],
            hex.v[4],
            hex.v[5],
            hex.v[6],
            hex.v[7],
        };

        glm::dvec3 vp[] = {
            verts[vi[0]],
            verts[vi[1]],
            verts[vi[2]],
            verts[vi[3]],
            verts[vi[4]],
            verts[vi[5]],
            verts[vi[6]],
            verts[vi[7]],
        };

        glm::dvec3 oct[] = {
            (vp[0] + vp[1] + vp[2] + vp[3]) / 4.0,
            (vp[0] + vp[1] + vp[4] + vp[5]) / 4.0,
            (vp[1] + vp[3] + vp[5] + vp[7]) / 4.0,
            (vp[2] + vp[3] + vp[6] + vp[7]) / 4.0,
            (vp[0] + vp[2] + vp[4] + vp[6]) / 4.0,
            (vp[4] + vp[5] + vp[6] + vp[7]) / 4.0,
        };

        glm::dvec3 n[] = {
            glm::cross(oct[1] - oct[0], oct[4] - oct[0]),
            glm::cross(oct[2] - oct[0], oct[1] - oct[0]),
            glm::cross(oct[4] - oct[0], oct[3] - oct[0]),
            glm::cross(oct[3] - oct[0], oct[2] - oct[0]),
            glm::cross(oct[4] - oct[5], oct[1] - oct[5]),
            glm::cross(oct[1] - oct[5], oct[2] - oct[5]),
            glm::cross(oct[3] - oct[5], oct[4] - oct[5]),
            glm::cross(oct[2] - oct[5], oct[3] - oct[5]),
        };

        glm::dvec3 c[] = {
            (oct[0] + oct[1] + oct[4]) / 3.0,
            (oct[0] + oct[1] + oct[2]) / 3.0,
            (oct[0] + oct[3] + oct[4]) / 3.0,
            (oct[0] + oct[2] + oct[3]) / 3.0,
            (oct[1] + oct[4] + oct[5]) / 3.0,
            (oct[1] + oct[2] + oct[5]) / 3.0,
            (oct[3] + oct[4] + oct[5]) / 3.0,
            (oct[2] + oct[3] + oct[5]) / 3.0,
        };

        double quality = evaluator.hexQuality(vp);

        vp[0] = c[0] + _lambda * n[0] / glm::sqrt(glm::length(n[0]));
        vp[1] = c[1] + _lambda * n[1] / glm::sqrt(glm::length(n[1]));
        vp[2] = c[2] + _lambda * n[2] / glm::sqrt(glm::length(n[2]));
        vp[3] = c[3] + _lambda * n[3] / glm::sqrt(glm::length(n[3]));
        vp[4] = c[4] + _lambda * n[4] / glm::sqrt(glm::length(n[4]));
        vp[5] = c[5] + _lambda * n[5] / glm::sqrt(glm::length(n[5]));
        vp[6] = c[6] + _lambda * n[6] / glm::sqrt(glm::length(n[6]));
        vp[7] = c[7] + _lambda * n[7] / glm::sqrt(glm::length(n[7]));

        if(topos[vi[0]].isBoundary) vp[0] = (*topos[vi[0]].snapToBoundary)(vp[0]);
        if(topos[vi[1]].isBoundary) vp[1] = (*topos[vi[1]].snapToBoundary)(vp[1]);
        if(topos[vi[2]].isBoundary) vp[2] = (*topos[vi[2]].snapToBoundary)(vp[2]);
        if(topos[vi[3]].isBoundary) vp[3] = (*topos[vi[3]].snapToBoundary)(vp[3]);
        if(topos[vi[4]].isBoundary) vp[4] = (*topos[vi[4]].snapToBoundary)(vp[4]);
        if(topos[vi[5]].isBoundary) vp[5] = (*topos[vi[5]].snapToBoundary)(vp[5]);
        if(topos[vi[6]].isBoundary) vp[6] = (*topos[vi[6]].snapToBoundary)(vp[6]);
        if(topos[vi[7]].isBoundary) vp[7] = (*topos[vi[7]].snapToBoundary)(vp[7]);

        double qualityPrime = evaluator.hexQuality(vp);
        double weight = qualityPrime / quality;

        _vertexAccums[vi[0]]->add(vp[0], weight);
        _vertexAccums[vi[1]]->add(vp[1], weight);
        _vertexAccums[vi[2]]->add(vp[2], weight);
        _vertexAccums[vi[3]]->add(vp[3], weight);
        _vertexAccums[vi[4]]->add(vp[4], weight);
        _vertexAccums[vi[5]]->add(vp[5], weight);
        _vertexAccums[vi[6]]->add(vp[6], weight);
        _vertexAccums[vi[7]]->add(vp[7], weight);
    }
}
