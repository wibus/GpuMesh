#include "VolumeEdgeEvaluator.h"

using namespace glm;


VolumeEdgeEvaluator::VolumeEdgeEvaluator() :
    AbstractEvaluator(":/shaders/compute/Quality/VolumeEdge.glsl")
{
    // This algorithm seems to be numerically unstable.
    // That is wy we use min(quality, 1.0) on the final result.
    // The instability must come from the sum of
    // determinants and edge's dot products.
}

VolumeEdgeEvaluator::~VolumeEdgeEvaluator()
{

}

double VolumeEdgeEvaluator::tetQuality(const dvec3 verts[]) const
{
    double volume = 0.0;
    for(int t=0; t < MeshTet::TET_COUNT; ++t)
    {
        volume += determinant(dmat3(
            verts[MeshTet::tets[t][0]] - verts[MeshTet::tets[t][3]],
            verts[MeshTet::tets[t][1]] - verts[MeshTet::tets[t][3]],
            verts[MeshTet::tets[t][2]] - verts[MeshTet::tets[t][3]]));
    }

    double edge2Sum = 0.0;
    for(int e=0; e < MeshTet::EDGE_COUNT; ++e)
    {
        dvec3 edge = verts[MeshTet::edges[e][0]] -
                     verts[MeshTet::edges[e][1]];
        edge2Sum += dot(edge, edge);
    }

    return min(volume / (edge2Sum*sqrt(edge2Sum))
            / 0.048112522432468815548, 1.0); // Normalization constant
}

double VolumeEdgeEvaluator::priQuality(const dvec3 verts[]) const
{
    double volume = 0.0;
    for(int t=0; t < MeshPri::TET_COUNT; ++t)
    {
        volume += determinant(dmat3(
            verts[MeshPri::tets[t][0]] - verts[MeshPri::tets[t][3]],
            verts[MeshPri::tets[t][1]] - verts[MeshPri::tets[t][3]],
            verts[MeshPri::tets[t][2]] - verts[MeshPri::tets[t][3]]));
    }

    double edge2Sum = 0.0;
    for(int e=0; e < MeshPri::EDGE_COUNT; ++e)
    {
        dvec3 edge = verts[MeshPri::edges[e][0]] -
                     verts[MeshPri::edges[e][1]];
        edge2Sum += dot(edge, edge);
    }

    return min(volume / (edge2Sum*sqrt(edge2Sum))
            / 0.088228615568855695006, 1.0); // Normalization constant
}

double VolumeEdgeEvaluator::hexQuality(const dvec3 verts[]) const
{
    double volume = 0.0;
    for(int t=0; t < MeshHex::TET_COUNT; ++t)
    {
        volume += determinant(dmat3(
            verts[MeshHex::tets[t][0]] - verts[MeshHex::tets[t][3]],
            verts[MeshHex::tets[t][1]] - verts[MeshHex::tets[t][3]],
            verts[MeshHex::tets[t][2]] - verts[MeshHex::tets[t][3]]));
    }

    double edge2Sum = 0.0;
    for(int e=0; e < MeshHex::EDGE_COUNT; ++e)
    {
        dvec3 edge = verts[MeshHex::edges[e][0]] -
                     verts[MeshHex::edges[e][1]];
        edge2Sum += dot(edge, edge);
    }

    return min(volume / (edge2Sum*sqrt(edge2Sum))
            / 0.14433756729740643276, 1.0); // Normalization constant
}
