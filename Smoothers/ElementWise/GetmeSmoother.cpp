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

        glm::dvec3 vpp[] = {
            vp[0] + n[0] * (_lambda / glm::sqrt(glm::length(n[0]))),
            vp[1] + n[1] * (_lambda / glm::sqrt(glm::length(n[1]))),
            vp[2] + n[2] * (_lambda / glm::sqrt(glm::length(n[2]))),
            vp[3] + n[3] * (_lambda / glm::sqrt(glm::length(n[3]))),
        };


        double volume =
            glm::determinant(
                glm::dmat3(
                    vp[0] - vp[3],
                    vp[1] - vp[3],
                    vp[2] - vp[3]));

        double volumePrime =
            glm::determinant(
                glm::dmat3(
                    vpp[0] - vpp[3],
                    vpp[1] - vpp[3],
                    vpp[2] - vpp[3]));

        double absVolumeRation = glm::abs(volume / volumePrime);
        double volumeVar = glm::pow(absVolumeRation, 1.0/3.0);

        vpp[0] = center + volumeVar * (vpp[0] - center);
        vpp[1] = center + volumeVar * (vpp[1] - center);
        vpp[2] = center + volumeVar * (vpp[2] - center);
        vpp[3] = center + volumeVar * (vpp[3] - center);

        if(topos[vi[0]].isBoundary) vpp[0] = (*topos[vi[0]].snapToBoundary)(vpp[0]);
        if(topos[vi[1]].isBoundary) vpp[1] = (*topos[vi[1]].snapToBoundary)(vpp[1]);
        if(topos[vi[2]].isBoundary) vpp[2] = (*topos[vi[2]].snapToBoundary)(vpp[2]);
        if(topos[vi[3]].isBoundary) vpp[3] = (*topos[vi[3]].snapToBoundary)(vpp[3]);

        double quality = evaluator.tetQuality(vp);
        double qualityPrime = evaluator.tetQuality(vpp);
        double weight = qualityPrime / quality;

        _vertexAccums[tet[0]]->add(vpp[0], weight);
        _vertexAccums[tet[1]]->add(vpp[1], weight);
        _vertexAccums[tet[2]]->add(vpp[2], weight);
        _vertexAccums[tet[3]]->add(vpp[3], weight);
    }
}

void GetmeSmoother::smoothPris(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last)
{
    const vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;
    const vector<MeshPri>& pris = mesh.pris;

    for(int e = first; e < last; ++e)
    {
        const MeshPri& pri = pris[e];

        uint vi[] = {
            pri.v[0],
            pri.v[1],
            pri.v[2],
            pri.v[3],
            pri.v[4],
            pri.v[5],
        };

        glm::dvec3 vp[] = {
            verts[vi[0]],
            verts[vi[1]],
            verts[vi[2]],
            verts[vi[3]],
            verts[vi[4]],
            verts[vi[5]],
        };

        glm::dvec3 aux[] = {
            (vp[0] + vp[2] + vp[4]) / 3.0,
            (vp[0] + vp[1] + vp[4] + vp[5]) / 4.0,
            (vp[0] + vp[1] + vp[2] + vp[3]) / 4.0,
            (vp[2] + vp[3] + vp[4] + vp[5]) / 4.0,
            (vp[1] + vp[3] + vp[5]) / 3.0,
        };

        glm::dvec3 n[] = {
            glm::cross(aux[2] - aux[0], aux[1] - aux[0]),
            glm::cross(aux[1] - aux[4], aux[2] - aux[4]),
            glm::cross(aux[3] - aux[0], aux[2] - aux[0]),
            glm::cross(aux[2] - aux[4], aux[3] - aux[4]),
            glm::cross(aux[1] - aux[0], aux[3] - aux[0]),
            glm::cross(aux[3] - aux[4], aux[1] - aux[4]),
        };

        double t = (4.0/5.0) * (1.0 - glm::pow(4.0/39.0, 0.25) * _lambda);
        double it = 1.0 - t;
        glm::dvec3 bases[] = {
            it * aux[0] + t * (aux[1] + aux[2]) / 2.0,
            it * aux[4] + t * (aux[1] + aux[2]) / 2.0,
            it * aux[0] + t * (aux[2] + aux[3]) / 2.0,
            it * aux[4] + t * (aux[2] + aux[3]) / 2.0,
            it * aux[0] + t * (aux[1] + aux[3]) / 2.0,
            it * aux[4] + t * (aux[1] + aux[3]) / 2.0,
        };


        // New positions
        glm::dvec3 vpp[] = {
            bases[0] + n[0] * (_lambda / glm::sqrt(glm::length(n[0]))),
            bases[1] + n[1] * (_lambda / glm::sqrt(glm::length(n[1]))),
            bases[2] + n[2] * (_lambda / glm::sqrt(glm::length(n[2]))),
            bases[3] + n[3] * (_lambda / glm::sqrt(glm::length(n[3]))),
            bases[4] + n[4] * (_lambda / glm::sqrt(glm::length(n[4]))),
            bases[5] + n[5] * (_lambda / glm::sqrt(glm::length(n[5]))),
        };

        if(topos[vi[0]].isBoundary) vpp[0] = (*topos[vi[0]].snapToBoundary)(vpp[0]);
        if(topos[vi[1]].isBoundary) vpp[1] = (*topos[vi[1]].snapToBoundary)(vpp[1]);
        if(topos[vi[2]].isBoundary) vpp[2] = (*topos[vi[2]].snapToBoundary)(vpp[2]);
        if(topos[vi[3]].isBoundary) vpp[3] = (*topos[vi[3]].snapToBoundary)(vpp[3]);
        if(topos[vi[4]].isBoundary) vpp[4] = (*topos[vi[4]].snapToBoundary)(vpp[4]);
        if(topos[vi[5]].isBoundary) vpp[5] = (*topos[vi[5]].snapToBoundary)(vpp[5]);


        double quality = evaluator.priQuality(vp);
        double qualityPrime = evaluator.priQuality(vpp);
        double weight = qualityPrime / quality;

        _vertexAccums[vi[0]]->add(vpp[0], weight);
        _vertexAccums[vi[1]]->add(vpp[1], weight);
        _vertexAccums[vi[2]]->add(vpp[2], weight);
        _vertexAccums[vi[3]]->add(vpp[3], weight);
        _vertexAccums[vi[4]]->add(vpp[4], weight);
        _vertexAccums[vi[5]]->add(vpp[5], weight);
    }
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

        glm::dvec3 aux[] = {
            (vp[0] + vp[1] + vp[2] + vp[3]) / 4.0,
            (vp[0] + vp[1] + vp[4] + vp[5]) / 4.0,
            (vp[1] + vp[3] + vp[5] + vp[7]) / 4.0,
            (vp[2] + vp[3] + vp[6] + vp[7]) / 4.0,
            (vp[0] + vp[2] + vp[4] + vp[6]) / 4.0,
            (vp[4] + vp[5] + vp[6] + vp[7]) / 4.0,
        };

        glm::dvec3 n[] = {
            glm::cross(aux[1] - aux[0], aux[4] - aux[0]),
            glm::cross(aux[2] - aux[0], aux[1] - aux[0]),
            glm::cross(aux[4] - aux[0], aux[3] - aux[0]),
            glm::cross(aux[3] - aux[0], aux[2] - aux[0]),
            glm::cross(aux[4] - aux[5], aux[1] - aux[5]),
            glm::cross(aux[1] - aux[5], aux[2] - aux[5]),
            glm::cross(aux[3] - aux[5], aux[4] - aux[5]),
            glm::cross(aux[2] - aux[5], aux[3] - aux[5]),
        };

        glm::dvec3 bases[] = {
            (aux[0] + aux[1] + aux[4]) / 3.0,
            (aux[0] + aux[1] + aux[2]) / 3.0,
            (aux[0] + aux[3] + aux[4]) / 3.0,
            (aux[0] + aux[2] + aux[3]) / 3.0,
            (aux[1] + aux[4] + aux[5]) / 3.0,
            (aux[1] + aux[2] + aux[5]) / 3.0,
            (aux[3] + aux[4] + aux[5]) / 3.0,
            (aux[2] + aux[3] + aux[5]) / 3.0,
        };


        // New positions
        glm::dvec3 vpp[] = {
            bases[0] + n[0] * (_lambda / glm::sqrt(glm::length(n[0]))),
            bases[1] + n[1] * (_lambda / glm::sqrt(glm::length(n[1]))),
            bases[2] + n[2] * (_lambda / glm::sqrt(glm::length(n[2]))),
            bases[3] + n[3] * (_lambda / glm::sqrt(glm::length(n[3]))),
            bases[4] + n[4] * (_lambda / glm::sqrt(glm::length(n[4]))),
            bases[5] + n[5] * (_lambda / glm::sqrt(glm::length(n[5]))),
            bases[6] + n[6] * (_lambda / glm::sqrt(glm::length(n[6]))),
            bases[7] + n[7] * (_lambda / glm::sqrt(glm::length(n[7]))),
        };

        if(topos[vi[0]].isBoundary) vpp[0] = (*topos[vi[0]].snapToBoundary)(vpp[0]);
        if(topos[vi[1]].isBoundary) vpp[1] = (*topos[vi[1]].snapToBoundary)(vpp[1]);
        if(topos[vi[2]].isBoundary) vpp[2] = (*topos[vi[2]].snapToBoundary)(vpp[2]);
        if(topos[vi[3]].isBoundary) vpp[3] = (*topos[vi[3]].snapToBoundary)(vpp[3]);
        if(topos[vi[4]].isBoundary) vpp[4] = (*topos[vi[4]].snapToBoundary)(vpp[4]);
        if(topos[vi[5]].isBoundary) vpp[5] = (*topos[vi[5]].snapToBoundary)(vpp[5]);
        if(topos[vi[6]].isBoundary) vpp[6] = (*topos[vi[6]].snapToBoundary)(vpp[6]);
        if(topos[vi[7]].isBoundary) vpp[7] = (*topos[vi[7]].snapToBoundary)(vpp[7]);


        double quality = evaluator.hexQuality(vp);
        double qualityPrime = evaluator.hexQuality(vp);
        double weight = qualityPrime / quality;

        _vertexAccums[vi[0]]->add(vpp[0], weight);
        _vertexAccums[vi[1]]->add(vpp[1], weight);
        _vertexAccums[vi[2]]->add(vpp[2], weight);
        _vertexAccums[vi[3]]->add(vpp[3], weight);
        _vertexAccums[vi[4]]->add(vpp[4], weight);
        _vertexAccums[vi[5]]->add(vpp[5], weight);
        _vertexAccums[vi[6]]->add(vpp[6], weight);
        _vertexAccums[vi[7]]->add(vpp[7], weight);
    }
}
