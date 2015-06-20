#include "Mesh.h"

#include <algorithm>
#include <iostream>

using namespace std;






unsigned int Mesh::vertCount() const
{
    return vert.size();
}

unsigned int Mesh::elemCount() const
{
    return tetra.size() * 12;
}

double Mesh::tetrahedronQuality(const glm::ivec4& tet)
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

void Mesh::compileTetrahedronQuality(
        double& qualityMean,
        double& qualityVar)
{
    qualityMean = 0.0;
    qualityVar = 0.0;

    int tetCount = tetra.size();
    for(int i=0; i < tetCount; ++i)
    {
        const glm::ivec4& tet = tetra[i];
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

    int tetCount = tetra.size();
    for(int i=0; i < tetCount; ++i)
    {
        const glm::ivec4& tet = tetra[i];

        glm::dvec3 verts[] = {
            glm::vec3(vert[tet[0]]),
            glm::vec3(vert[tet[1]]),
            glm::vec3(vert[tet[2]]),
            glm::vec3(vert[tet[3]])
        };

        if(glm::dot(verts[0], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[1], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[2], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[3], cutNormal) - cutDistance > 0.0)
            continue;

        glm::dvec3 norms[] = {
            glm::normalize(glm::cross(verts[1] - verts[0], verts[2] - verts[1])),
            glm::normalize(glm::cross(verts[2] - verts[0], verts[3] - verts[2])),
            glm::normalize(glm::cross(verts[3] - verts[0], verts[1] - verts[3])),
            glm::normalize(glm::cross(verts[3] - verts[1], verts[2] - verts[3])),
        };

        double quality = tetrahedronQuality(tet);

        pushTriangle(vertices, normals, triEdges, qualities,
                     verts[0], verts[1], verts[2], norms[0], quality);
        pushTriangle(vertices, normals, triEdges, qualities,
                     verts[0], verts[2], verts[3], norms[1], quality);
        pushTriangle(vertices, normals, triEdges, qualities,
                     verts[0], verts[3], verts[1], norms[2], quality);
        pushTriangle(vertices, normals, triEdges, qualities,
                     verts[1], verts[3], verts[2], norms[3], quality);
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
