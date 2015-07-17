#include "VolumeEdgeEvaluator.h"

using namespace glm;


VolumeEdgeEvaluator::VolumeEdgeEvaluator() :
    AbstractEvaluator(":/shaders/compute/Quality/VolumeEdge.glsl")
{

}

VolumeEdgeEvaluator::~VolumeEdgeEvaluator()
{

}

double VolumeEdgeEvaluator::tetrahedronQuality(const dvec3 verts[]) const
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

    return volume / (edge2Sum*sqrt(edge2Sum))
             * 20.7846096908; // Normalization constant
}

double VolumeEdgeEvaluator::prismQuality(const dvec3 verts[]) const
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

    return volume / (edge2Sum*sqrt(edge2Sum))
             * 11.3341912208; // Normalization constant
}

double VolumeEdgeEvaluator::hexahedronQuality(const dvec3 verts[]) const
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

    return volume / (edge2Sum*sqrt(edge2Sum))
            * 10.3923048454; // Normalization constant
}
