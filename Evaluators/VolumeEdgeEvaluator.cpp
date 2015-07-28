#include "VolumeEdgeEvaluator.h"

#include <GLM/gtx/norm.hpp>

using namespace glm;


VolumeEdgeEvaluator::VolumeEdgeEvaluator() :
    AbstractEvaluator(":/shaders/compute/Quality/VolumeEdge.glsl")
{
    // This algorithm seems to be numerically unstable.
    // That is why we use min(quality, 1.0) on the final result.
    // The instability must come from the sums of
    // determinants and edge's dot products.
}

VolumeEdgeEvaluator::~VolumeEdgeEvaluator()
{

}

double VolumeEdgeEvaluator::tetQuality(const dvec3 vp[]) const
{
    double volume = determinant(dmat3(
        vp[MeshTet::tets[0][0]] - vp[MeshTet::tets[0][3]],
        vp[MeshTet::tets[0][1]] - vp[MeshTet::tets[0][3]],
        vp[MeshTet::tets[0][2]] - vp[MeshTet::tets[0][3]]));

    double edge2Sum = 0.0;
    edge2Sum += length2(vp[MeshTet::edges[0][0]] - vp[MeshTet::edges[0][1]]);
    edge2Sum += length2(vp[MeshTet::edges[1][0]] - vp[MeshTet::edges[1][1]]);
    edge2Sum += length2(vp[MeshTet::edges[2][0]] - vp[MeshTet::edges[2][1]]);
    edge2Sum += length2(vp[MeshTet::edges[3][0]] - vp[MeshTet::edges[3][1]]);
    edge2Sum += length2(vp[MeshTet::edges[4][0]] - vp[MeshTet::edges[4][1]]);
    edge2Sum += length2(vp[MeshTet::edges[5][0]] - vp[MeshTet::edges[5][1]]);

    return min(volume / (edge2Sum*sqrt(edge2Sum))
            / 0.048112522432468815548, 1.0); // Normalization constant
}

double VolumeEdgeEvaluator::priQuality(const dvec3 vp[]) const
{
    double volume = 0.0;
    volume += determinant(dmat3(
        vp[MeshPri::tets[0][0]] - vp[MeshPri::tets[0][3]],
        vp[MeshPri::tets[0][1]] - vp[MeshPri::tets[0][3]],
        vp[MeshPri::tets[0][2]] - vp[MeshPri::tets[0][3]]));
    volume += determinant(dmat3(
        vp[MeshPri::tets[1][0]] - vp[MeshPri::tets[1][3]],
        vp[MeshPri::tets[1][1]] - vp[MeshPri::tets[1][3]],
        vp[MeshPri::tets[1][2]] - vp[MeshPri::tets[1][3]]));
    volume += determinant(dmat3(
        vp[MeshPri::tets[2][0]] - vp[MeshPri::tets[2][3]],
        vp[MeshPri::tets[2][1]] - vp[MeshPri::tets[2][3]],
        vp[MeshPri::tets[2][2]] - vp[MeshPri::tets[2][3]]));

    double edge2Sum = 0.0;
    edge2Sum += length2(vp[MeshPri::edges[0][0]] - vp[MeshPri::edges[0][1]]);
    edge2Sum += length2(vp[MeshPri::edges[1][0]] - vp[MeshPri::edges[1][1]]);
    edge2Sum += length2(vp[MeshPri::edges[2][0]] - vp[MeshPri::edges[2][1]]);
    edge2Sum += length2(vp[MeshPri::edges[3][0]] - vp[MeshPri::edges[3][1]]);
    edge2Sum += length2(vp[MeshPri::edges[4][0]] - vp[MeshPri::edges[4][1]]);
    edge2Sum += length2(vp[MeshPri::edges[5][0]] - vp[MeshPri::edges[5][1]]);
    edge2Sum += length2(vp[MeshPri::edges[6][0]] - vp[MeshPri::edges[6][1]]);
    edge2Sum += length2(vp[MeshPri::edges[7][0]] - vp[MeshPri::edges[7][1]]);
    edge2Sum += length2(vp[MeshPri::edges[8][0]] - vp[MeshPri::edges[8][1]]);

    return min(volume / (edge2Sum*sqrt(edge2Sum))
            / 0.096225044864937631095, 1.0); // Normalization constant
}

double VolumeEdgeEvaluator::hexQuality(const dvec3 vp[]) const
{
    double volume = 0.0;
    volume += determinant(dmat3(
        vp[MeshHex::tets[0][0]] - vp[MeshHex::tets[0][3]],
        vp[MeshHex::tets[0][1]] - vp[MeshHex::tets[0][3]],
        vp[MeshHex::tets[0][2]] - vp[MeshHex::tets[0][3]]));
    volume += determinant(dmat3(
        vp[MeshHex::tets[1][0]] - vp[MeshHex::tets[1][3]],
        vp[MeshHex::tets[1][1]] - vp[MeshHex::tets[1][3]],
        vp[MeshHex::tets[1][2]] - vp[MeshHex::tets[1][3]]));
    volume += determinant(dmat3(
        vp[MeshHex::tets[2][0]] - vp[MeshHex::tets[2][3]],
        vp[MeshHex::tets[2][1]] - vp[MeshHex::tets[2][3]],
        vp[MeshHex::tets[2][2]] - vp[MeshHex::tets[2][3]]));
    volume += determinant(dmat3(
        vp[MeshHex::tets[3][0]] - vp[MeshHex::tets[3][3]],
        vp[MeshHex::tets[3][1]] - vp[MeshHex::tets[3][3]],
        vp[MeshHex::tets[3][2]] - vp[MeshHex::tets[3][3]]));
    volume += determinant(dmat3(
        vp[MeshHex::tets[4][0]] - vp[MeshHex::tets[4][3]],
        vp[MeshHex::tets[4][1]] - vp[MeshHex::tets[4][3]],
        vp[MeshHex::tets[4][2]] - vp[MeshHex::tets[4][3]]));

    double edge2Sum = 0.0;
    edge2Sum += length2(vp[MeshHex::edges[0][0]] - vp[MeshHex::edges[0][1]]);
    edge2Sum += length2(vp[MeshHex::edges[1][0]] - vp[MeshHex::edges[1][1]]);
    edge2Sum += length2(vp[MeshHex::edges[2][0]] - vp[MeshHex::edges[2][1]]);
    edge2Sum += length2(vp[MeshHex::edges[3][0]] - vp[MeshHex::edges[3][1]]);
    edge2Sum += length2(vp[MeshHex::edges[4][0]] - vp[MeshHex::edges[4][1]]);
    edge2Sum += length2(vp[MeshHex::edges[5][0]] - vp[MeshHex::edges[5][1]]);
    edge2Sum += length2(vp[MeshHex::edges[6][0]] - vp[MeshHex::edges[6][1]]);
    edge2Sum += length2(vp[MeshHex::edges[7][0]] - vp[MeshHex::edges[7][1]]);
    edge2Sum += length2(vp[MeshHex::edges[8][0]] - vp[MeshHex::edges[8][1]]);
    edge2Sum += length2(vp[MeshHex::edges[9][0]] - vp[MeshHex::edges[9][1]]);
    edge2Sum += length2(vp[MeshHex::edges[10][0]] - vp[MeshHex::edges[10][1]]);
    edge2Sum += length2(vp[MeshHex::edges[11][0]] - vp[MeshHex::edges[11][1]]);

    return min(volume / (edge2Sum*sqrt(edge2Sum))
            / 0.14433756729740643276, 1.0); // Normalization constant
}
