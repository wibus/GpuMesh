#ifndef GPUMESH_TETRAHEDRALIZER
#define GPUMESH_TETRAHEDRALIZER

#include <vector>

#include "Mesh.h"


const uint PRI_ROT[6][6] = {
    {0, 1, 2, 3, 4, 5},
    {1, 2, 0, 4, 5, 3},
    {2, 0, 1, 5, 3, 4},
    {3, 5, 4, 0, 2, 1},
    {4, 3, 5, 1, 0, 2},
    {5, 4, 3, 2, 1, 0}
};

const uint HEX_ROT[8][8] = {
    {0, 1, 2, 3, 4, 5, 6, 7},
    {1, 0, 4, 5, 2, 3, 7, 6},
    {2, 1, 5, 6, 3, 0, 4, 7},
    {3, 0, 1, 2, 7, 4, 5, 6},
    {4, 0, 3, 7, 5, 1, 2, 6},
    {5, 1, 0, 4, 6, 2, 3, 7},
    {6, 2, 1, 5, 7, 3, 0, 4},
    {7, 3, 2, 6, 4, 0, 1, 5}
};

const uint HEX_LOOP[8] = {
    0, 120, 240, 0, 0, 240, 120, 0
};

const uint HEX_F0 = 0b100;
const uint HEX_F1 = 0b010;
const uint HEX_F2 = 0b001;


template<typename Tet>
void tetrahedrize(
        std::vector<Tet>& tets,
        const Mesh& mesh)
{
    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    size_t maxTetCount = tetCount +
        priCount * 3 + hexCount * 6;

    tets.reserve(maxTetCount);


    // Tets
    for(const MeshTet& tet : mesh.tets)
        tets.push_back(Tet(tet));


    // Prisms
    for(const MeshPri& pri : mesh.pris)
    {
        uint smallestId = 0;
        if(pri.v[1] < pri.v[smallestId]) smallestId = 1;
        if(pri.v[2] < pri.v[smallestId]) smallestId = 2;
        if(pri.v[3] < pri.v[smallestId]) smallestId = 3;
        if(pri.v[4] < pri.v[smallestId]) smallestId = 4;
        if(pri.v[5] < pri.v[smallestId]) smallestId = 5;

        uint vIds[6] = {
            pri.v[PRI_ROT[smallestId][0]],
            pri.v[PRI_ROT[smallestId][1]],
            pri.v[PRI_ROT[smallestId][2]],
            pri.v[PRI_ROT[smallestId][3]],
            pri.v[PRI_ROT[smallestId][4]],
            pri.v[PRI_ROT[smallestId][5]],
        };

        if(glm::min(vIds[1], vIds[5]) < glm::min(vIds[2], vIds[4]))
        {
            tets.push_back(Tet(vIds[0], vIds[1], vIds[2], vIds[5]));
            tets.push_back(Tet(vIds[0], vIds[1], vIds[5], vIds[4]));
            tets.push_back(Tet(vIds[0], vIds[4], vIds[5], vIds[3]));
        }
        else
        {
            tets.push_back(Tet(vIds[0], vIds[1], vIds[2], vIds[4]));
            tets.push_back(Tet(vIds[0], vIds[4], vIds[2], vIds[5]));
            tets.push_back(Tet(vIds[0], vIds[4], vIds[5], vIds[3]));
        }
    }


    // Hexs
    for(const MeshHex& hex : mesh.hexs)
    {
        uint smallestId = 0;
        if(hex.v[1] < hex.v[smallestId]) smallestId = 1;
        if(hex.v[2] < hex.v[smallestId]) smallestId = 2;
        if(hex.v[3] < hex.v[smallestId]) smallestId = 3;
        if(hex.v[4] < hex.v[smallestId]) smallestId = 4;
        if(hex.v[5] < hex.v[smallestId]) smallestId = 5;
        if(hex.v[6] < hex.v[smallestId]) smallestId = 6;
        if(hex.v[7] < hex.v[smallestId]) smallestId = 7;

        uint vIds[8] = {
            hex.v[HEX_ROT[smallestId][0]],
            hex.v[HEX_ROT[smallestId][1]],
            hex.v[HEX_ROT[smallestId][2]],
            hex.v[HEX_ROT[smallestId][3]],
            hex.v[HEX_ROT[smallestId][4]],
            hex.v[HEX_ROT[smallestId][5]],
            hex.v[HEX_ROT[smallestId][6]],
            hex.v[HEX_ROT[smallestId][7]]
        };

        uint diag = 0;
        if(glm::min(vIds[1], vIds[6]) < glm::min(vIds[2], vIds[5]))
        {
            diag |= HEX_F0;
        }
        if(glm::min(vIds[3], vIds[6]) < glm::min(vIds[2], vIds[7]))
        {
            diag |= HEX_F1;
        }
        if(glm::min(vIds[4], vIds[6]) < glm::min(vIds[5], vIds[7]))
        {
            diag |= HEX_F2;
        }

        uint tmp;
        switch(HEX_LOOP[diag])
        {
        case 0 : break; // No rotation
        case 120:
            tmp = vIds[1]; vIds[1] = vIds[4]; vIds[4] = vIds[3]; vIds[3] = tmp;
            tmp = vIds[5]; vIds[5] = vIds[7]; vIds[7] = vIds[2]; vIds[2] = tmp;
            break;

        case 240:
            tmp = vIds[1]; vIds[1] = vIds[3]; vIds[3] = vIds[4]; vIds[4] = tmp;
            tmp = vIds[5]; vIds[5] = vIds[2]; vIds[2] = vIds[7]; vIds[7] = tmp;
            break;

        default:
            assert(false /* Hex can only be rotated by 120 or 240 degrees */);
        }

        uint sum = (diag & HEX_F2) + ((diag & HEX_F1) >> 1) + ((diag & HEX_F0) >> 2);
        switch(sum)
        {
        case 0 :
            tets.push_back(Tet(vIds[0], vIds[1], vIds[2], vIds[5]));
            tets.push_back(Tet(vIds[0], vIds[2], vIds[7], vIds[5]));
            tets.push_back(Tet(vIds[0], vIds[2], vIds[3], vIds[7]));
            tets.push_back(Tet(vIds[0], vIds[5], vIds[7], vIds[4]));
            tets.push_back(Tet(vIds[2], vIds[7], vIds[5], vIds[6]));
            break;

        case 1 :
            tets.push_back(Tet(vIds[0], vIds[5], vIds[7], vIds[4]));
            tets.push_back(Tet(vIds[0], vIds[1], vIds[7], vIds[5]));
            tets.push_back(Tet(vIds[1], vIds[6], vIds[7], vIds[5]));
            tets.push_back(Tet(vIds[0], vIds[7], vIds[2], vIds[3]));
            tets.push_back(Tet(vIds[0], vIds[7], vIds[1], vIds[2]));
            tets.push_back(Tet(vIds[1], vIds[7], vIds[6], vIds[2]));
            break;

        case 2 :
            tets.push_back(Tet(vIds[0], vIds[4], vIds[5], vIds[6]));
            tets.push_back(Tet(vIds[0], vIds[3], vIds[7], vIds[6]));
            tets.push_back(Tet(vIds[0], vIds[7], vIds[4], vIds[6]));
            tets.push_back(Tet(vIds[0], vIds[1], vIds[2], vIds[5]));
            tets.push_back(Tet(vIds[0], vIds[3], vIds[6], vIds[2]));
            tets.push_back(Tet(vIds[0], vIds[6], vIds[5], vIds[2]));
            break;

        case 3 :
            tets.push_back(Tet(vIds[0], vIds[2], vIds[3], vIds[6]));
            tets.push_back(Tet(vIds[0], vIds[3], vIds[7], vIds[6]));
            tets.push_back(Tet(vIds[0], vIds[7], vIds[4], vIds[6]));
            tets.push_back(Tet(vIds[0], vIds[5], vIds[6], vIds[4]));
            tets.push_back(Tet(vIds[1], vIds[5], vIds[6], vIds[0]));
            tets.push_back(Tet(vIds[1], vIds[6], vIds[2], vIds[0]));
            break;

        default:
            assert(false /* There are only 3 bits to be summed */);
        }
    }
}



template<typename Tet>
bool tetParams(
        const std::vector<MeshVert>& verts,
        const Tet &tet,
        const glm::dvec3& p,
        double coor[4])
{
    // ref : https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Barycentric_coordinates_on_tetrahedra

    const glm::dvec3& vp0 = verts[tet.v[0]].p;
    const glm::dvec3& vp1 = verts[tet.v[1]].p;
    const glm::dvec3& vp2 = verts[tet.v[2]].p;
    const glm::dvec3& vp3 = verts[tet.v[3]].p;

    glm::dmat3 T(vp0 - vp3, vp1 - vp3, vp2 - vp3);

    glm::dvec3 y = glm::inverse(T) * (p - vp3);
    coor[0] = y[0];
    coor[1] = y[1];
    coor[2] = y[2];
    coor[3] = 1.0 - (y[0] + y[1] + y[2]);

    const double EPSILON_IN = -1e-8;
    bool isIn = (coor[0] >= EPSILON_IN && coor[1] >= EPSILON_IN &&
                 coor[2] >= EPSILON_IN && coor[3] >= EPSILON_IN);
    return isIn;
}

#endif // GPUMESH_TETRAHEDRALIZER
