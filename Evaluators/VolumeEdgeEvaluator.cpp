#include "VolumeEdgeEvaluator.h"

using namespace glm;


VolumeEdgeEvaluator::VolumeEdgeEvaluator() :
    AbstractEvaluator(":/shaders/compute/Quality/VolumeEdge.glsl")
{

}

VolumeEdgeEvaluator::~VolumeEdgeEvaluator()
{

}

double VolumeEdgeEvaluator::tetrahedronQuality(
        const Mesh& mesh, const MeshTet& tet) const
{
    dvec3 ev[] = {
        mesh.vert[tet.v[0]],
        mesh.vert[tet.v[1]],
        mesh.vert[tet.v[2]],
        mesh.vert[tet.v[3]],
    };

    double volume = 0.0;
    for(int t=0; t < MeshTet::TET_COUNT; ++t)
    {
        volume += determinant(dmat3(
            ev[MeshTet::tets[t][0]] - ev[MeshTet::tets[t][3]],
            ev[MeshTet::tets[t][1]] - ev[MeshTet::tets[t][3]],
            ev[MeshTet::tets[t][2]] - ev[MeshTet::tets[t][3]]));
    }

    double edge2Sum = 0.0;
    for(int e=0; e < MeshTet::EDGE_COUNT; ++e)
    {
        dvec3 edge = ev[MeshTet::edges[e][0]] -
                     ev[MeshTet::edges[e][1]];
        edge2Sum += dot(edge, edge);
    }

    return volume / (edge2Sum*sqrt(edge2Sum))
             * 20.7846096908; // Normalization constant
}

double VolumeEdgeEvaluator::prismQuality(
        const Mesh& mesh, const MeshPri& pri) const
{
    dvec3 ev[] = {
        mesh.vert[pri.v[0]],
        mesh.vert[pri.v[1]],
        mesh.vert[pri.v[2]],
        mesh.vert[pri.v[3]],
        mesh.vert[pri.v[4]],
        mesh.vert[pri.v[5]]
    };

    double volume = 0.0;
    for(int t=0; t < MeshPri::TET_COUNT; ++t)
    {
        volume += determinant(dmat3(
            ev[MeshPri::tets[t][0]] - ev[MeshPri::tets[t][3]],
            ev[MeshPri::tets[t][1]] - ev[MeshPri::tets[t][3]],
            ev[MeshPri::tets[t][2]] - ev[MeshPri::tets[t][3]]));
    }

    double edge2Sum = 0.0;
    for(int e=0; e < MeshPri::EDGE_COUNT; ++e)
    {
        dvec3 edge = ev[MeshPri::edges[e][0]] -
                     ev[MeshPri::edges[e][1]];
        edge2Sum += dot(edge, edge);
    }

    return volume / (edge2Sum*sqrt(edge2Sum))
             * 11.3341912208; // Normalization constant
}

double VolumeEdgeEvaluator::hexahedronQuality(
        const Mesh& mesh, const MeshHex& hex) const
{
    dvec3 ev[] = {
        mesh.vert[hex.v[0]],
        mesh.vert[hex.v[1]],
        mesh.vert[hex.v[2]],
        mesh.vert[hex.v[3]],
        mesh.vert[hex.v[4]],
        mesh.vert[hex.v[5]],
        mesh.vert[hex.v[6]],
        mesh.vert[hex.v[7]]
    };

    double volume = 0.0;
    for(int t=0; t < MeshHex::TET_COUNT; ++t)
    {
        volume += determinant(dmat3(
            ev[MeshHex::tets[t][0]] - ev[MeshHex::tets[t][3]],
            ev[MeshHex::tets[t][1]] - ev[MeshHex::tets[t][3]],
            ev[MeshHex::tets[t][2]] - ev[MeshHex::tets[t][3]]));
    }

    double edge2Sum = 0.0;
    for(int e=0; e < MeshHex::EDGE_COUNT; ++e)
    {
        dvec3 edge = ev[MeshHex::edges[e][0]] -
                     ev[MeshHex::edges[e][1]];
        edge2Sum += dot(edge, edge);
    }

    return volume / (edge2Sum*sqrt(edge2Sum))
            * 10.3923048454; // Normalization constant
}
