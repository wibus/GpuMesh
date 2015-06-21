#include "Mesh.h"

#include <algorithm>
#include <iostream>

using namespace std;




const MeshTri MeshTet::faces[MeshTet::FACE_COUNT] = {
    MeshTri(0, 1, 2),
    MeshTri(0, 2, 3),
    MeshTri(0, 3, 1),
    MeshTri(1, 3, 2)
};

const MeshTri MeshPen::faces[MeshPen::FACE_COUNT] = {
    MeshTri(0, 2, 1), // Z neg face 0
    MeshTri(1, 2, 3), // Z neg face 1
    MeshTri(0, 1, 4), // Y neg face 0
    MeshTri(1, 5, 4), // Y neg face 1
    MeshTri(2, 4, 3), // Y pos face 0
    MeshTri(3, 4, 5), // Y pos face 1
    MeshTri(0, 4, 2), // YZ neg face
    MeshTri(1, 3, 5)  // YZ pos face
};

const MeshTri MeshHex::faces[MeshHex::FACE_COUNT] = {
    MeshTri(0, 2, 1), // Z neg face 0
    MeshTri(1, 2, 3), // Z pos face 1
    MeshTri(4, 5, 6), // Z pos face 0
    MeshTri(5, 7, 6), // Z pos face 1
    MeshTri(0, 1, 4), // Y neg face 0
    MeshTri(1, 5, 4), // Y neg face 1
    MeshTri(2, 7, 3), // Y pos face 0
    MeshTri(2, 6, 7), // Y pos face 1
    MeshTri(0, 4, 2), // X neg face 0
    MeshTri(2, 4, 6), // X neg face 1
    MeshTri(1, 3, 7), // X pos face 0
    MeshTri(1, 7, 5), // X pos face 1
};

double Mesh::tetrahedronQuality(const MeshTet& tet)
{
    glm::dvec3 A(vert[tet[0]]);
    glm::dvec3 B(vert[tet[1]]);
    glm::dvec3 C(vert[tet[2]]);
    glm::dvec3 D(vert[tet[3]]);
    std::vector<double> lengths {
        glm::distance(A, B),
        glm::distance(A, C),
        glm::distance(A, D),
        glm::distance(B, C),
        glm::distance(D, B),
        glm::distance(C, D)
    };

    double maxLen = 0;
    for(auto l : lengths)
        if(l > maxLen)
            maxLen = l;

    double u = lengths[0];
    double v = lengths[1];
    double w = lengths[2];
    double U = lengths[5];
    double V = lengths[4];
    double W = lengths[3];

    double Volume = 4*u*u*v*v*w*w;
    Volume -= u*u*pow(v*v+w*w-U*U,2);
    Volume -= v*v*pow(w*w+u*u-V*V,2);
    Volume -= w*w*pow(u*u+v*v-W*W,2);
    Volume += (v*v+w*w-U*U)*(w*w+u*u-V*V)*(u*u+v*v-W*W);
    Volume = sqrt(Volume);
    Volume /= 12;

    double s1 = (double) ((U + V + W) / 2);
    double s2 = (double) ((u + v + W) / 2);
    double s3 = (double) ((u + V + w) / 2);
    double s4 = (double) ((U + v + w) / 2);

    double L1 = sqrt(s1*(s1-U)*(s1-V)*(s1-W));
    double L2 = sqrt(s2*(s2-u)*(s2-v)*(s2-W));
    double L3 = sqrt(s3*(s3-u)*(s3-V)*(s3-w));
    double L4 = sqrt(s4*(s4-U)*(s4-v)*(s4-w));

    double R = (Volume*3)/(L1+L2+L3+L4);

    return (4.89897948557) * R / maxLen;
}


double Mesh::pentahedronQuality(const MeshPen& pen)
{
    return 1;
}

double Mesh::hexahedronQuality(const MeshHex& hex)
{
    return 1;
}

void Mesh::compileElementQuality(
        double& qualityMean,
        double& qualityVar)
{
    qualityMean = 0.0;
    qualityVar = 0.0;

    int tetCount = tetra.size();
    for(int i=0; i < tetCount; ++i)
    {
        const MeshTet& tet = tetra[i];
        double quality = tetrahedronQuality(tet);

        // Quality statistics
        qualityMean = (qualityMean * i + quality) / (i + 1);
        double qualityMeanDist = qualityMean - quality;
        double qualityMeanDist2 = qualityMeanDist*qualityMeanDist;
        qualityVar = (qualityVar * 1 + qualityMeanDist2) / (1 + 1);
    }
}

void Mesh::compileFacesAttributes(
        const glm::dvec4& cutPlaneEq,
        std::vector<glm::vec3>& vertices,
        std::vector<signed char>& normals,
        std::vector<unsigned char>& triEdges,
        std::vector<unsigned char>& qualities)
{
    glm::dvec3 cutNormal(cutPlaneEq);
    double cutDistance = cutPlaneEq.w;


    // Tetrahedrons
    int tetCount = tetra.size();
    for(int i=0; i < tetCount; ++i)
    {
        const MeshTet& tet = tetra[i];

        glm::dvec3 verts[] = {
            glm::dvec3(vert[tet[0]]),
            glm::dvec3(vert[tet[1]]),
            glm::dvec3(vert[tet[2]]),
            glm::dvec3(vert[tet[3]])
        };

        if(glm::dot(verts[0], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[1], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[2], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[3], cutNormal) - cutDistance > 0.0)
            continue;

        double quality = tetrahedronQuality(tet);


        for(int f=0; f < MeshTet::FACE_COUNT; ++f)
        {
            const MeshTri& tri = MeshTet::faces[f];
            glm::dvec3 A = verts[tri[1]] - verts[tri[0]];
            glm::dvec3 B = verts[tri[2]] - verts[tri[1]];
            glm::dvec3 normal = glm::normalize(glm::cross(A, B));
            pushTriangle(vertices, normals, triEdges, qualities,
                         verts[tri[0]], verts[tri[1]], verts[tri[2]],
                         normal, quality);
        }
    }


    // Pentahedrons
    int penCount = penta.size();
    for(int i=0; i < penCount; ++i)
    {
        const MeshPen& pen = penta[i];

        glm::dvec3 verts[] = {
            glm::dvec3(vert[pen[0]]),
            glm::dvec3(vert[pen[1]]),
            glm::dvec3(vert[pen[2]]),
            glm::dvec3(vert[pen[3]]),
            glm::dvec3(vert[pen[4]]),
            glm::dvec3(vert[pen[5]])
        };

        if(glm::dot(verts[0], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[1], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[2], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[3], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[4], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[5], cutNormal) - cutDistance > 0.0)
            continue;


        double quality = pentahedronQuality(pen);


        for(int f=0; f < MeshPen::FACE_COUNT; ++f)
        {
            const MeshTri& tri = MeshPen::faces[f];
            glm::dvec3 A = verts[tri[1]] - verts[tri[0]];
            glm::dvec3 B = verts[tri[2]] - verts[tri[1]];
            glm::dvec3 normal = glm::normalize(glm::cross(A, B));
            pushTriangle(vertices, normals, triEdges, qualities,
                         verts[tri[0]], verts[tri[1]], verts[tri[2]],
                         normal, quality);
        }
    }


    // Hexahedrons
    int hexCount = hexa.size();
    for(int i=0; i < hexCount; ++i)
    {
        const MeshHex& hex = hexa[i];

        glm::dvec3 verts[] = {
            glm::dvec3(vert[hex[0]]),
            glm::dvec3(vert[hex[1]]),
            glm::dvec3(vert[hex[2]]),
            glm::dvec3(vert[hex[3]]),
            glm::dvec3(vert[hex[4]]),
            glm::dvec3(vert[hex[5]]),
            glm::dvec3(vert[hex[6]]),
            glm::dvec3(vert[hex[7]])
        };

        if(glm::dot(verts[0], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[1], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[2], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[3], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[4], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[5], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[6], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[7], cutNormal) - cutDistance > 0.0)
            continue;


        double quality = hexahedronQuality(hex);


        for(int f=0; f < MeshHex::FACE_COUNT; ++f)
        {
            const MeshTri& tri = MeshHex::faces[f];
            glm::dvec3 A = verts[tri[1]] - verts[tri[0]];
            glm::dvec3 B = verts[tri[2]] - verts[tri[1]];
            glm::dvec3 normal = glm::normalize(glm::cross(A, B));
            pushTriangle(vertices, normals, triEdges, qualities,
                         verts[tri[0]], verts[tri[1]], verts[tri[2]],
                         normal, quality);
        }
    }
}

void Mesh::pushTriangle(
        std::vector<glm::vec3>& vertices,
        std::vector<signed char>& normals,
        std::vector<unsigned char>& triEdges,
        std::vector<unsigned char>& qualities,
        const glm::dvec3& A,
        const glm::dvec3& B,
        const glm::dvec3& C,
        const glm::dvec3& n,
        double quality)
{

    vertices.push_back(glm::vec3(A));
    vertices.push_back(glm::vec3(B));
    vertices.push_back(glm::vec3(C));

    signed char nx = n.x * 127;
    signed char ny = n.y * 127;
    signed char nz = n.z * 127;
    normals.push_back(nx);
    normals.push_back(ny);
    normals.push_back(nz);
    normals.push_back(nx);
    normals.push_back(ny);
    normals.push_back(nz);
    normals.push_back(nx);
    normals.push_back(ny);
    normals.push_back(nz);

    const glm::ivec3 X_EDGE(0,   255, 255);
    const glm::ivec3 Y_EDGE(255, 0,   255);
    const glm::ivec3 Z_EDGE(255, 255, 0  );
    triEdges.push_back(X_EDGE.x);
    triEdges.push_back(X_EDGE.y);
    triEdges.push_back(X_EDGE.z);
    triEdges.push_back(Y_EDGE.x);
    triEdges.push_back(Y_EDGE.y);
    triEdges.push_back(Y_EDGE.z);
    triEdges.push_back(Z_EDGE.x);
    triEdges.push_back(Z_EDGE.y);
    triEdges.push_back(Z_EDGE.z);

    qualities.push_back(quality * 255);
    qualities.push_back(quality * 255);
    qualities.push_back(quality * 255);
}
