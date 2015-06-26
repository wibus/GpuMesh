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

const int MeshTet::edges[MeshTet::EDGE_COUNT][2] = {
    {0, 1},
    {0, 2},
    {0, 3},
    {1, 2},
    {2, 3},
    {3, 1}
};


const MeshTri MeshPri::faces[MeshPri::FACE_COUNT] = {
    MeshTri(2, 1, 0), // Z neg face 0
    MeshTri(1, 2, 3), // Z neg face 1
    MeshTri(1, 4, 0), // Y neg face 0
    MeshTri(4, 1, 5), // Y neg face 1
    MeshTri(4, 3, 2), // Y pos face 0
    MeshTri(3, 4, 5), // Y pos face 1
    MeshTri(0, 4, 2), // X neg face
    MeshTri(1, 3, 5)  // X pos face
};

const int MeshPri::edges[MeshPri::EDGE_COUNT][2] = {
    {0, 1},
    {0, 2},
    {1, 3},
    {2, 3},
    {0, 4},
    {1, 5},
    {2, 4},
    {3, 5},
    {4, 5}
};


const MeshTri MeshHex::faces[MeshHex::FACE_COUNT] = {
    MeshTri(2, 1, 0), // Z neg face 0
    MeshTri(1, 2, 3), // Z pos face 1
    MeshTri(5, 6, 4), // Z pos face 0
    MeshTri(6, 5, 7), // Z pos face 1
    MeshTri(1, 4, 0), // Y neg face 0
    MeshTri(4, 1, 5), // Y neg face 1
    MeshTri(2, 7, 3), // Y pos face 0
    MeshTri(7, 2, 6), // Y pos face 1
    MeshTri(4, 2, 0), // X neg face 0
    MeshTri(2, 4, 6), // X neg face 1
    MeshTri(7, 1, 3), // X pos face 0
    MeshTri(1, 7, 5), // X pos face 1
};

const int MeshHex::edges[MeshHex::EDGE_COUNT][2] = {
    {0, 1},
    {0, 2},
    {1, 3},
    {2, 3},
    {0, 4},
    {1, 5},
    {2, 6},
    {3, 7},
    {4, 5},
    {4, 6},
    {5, 7},
    {6, 7}
};


MeshBound::MeshBound(int id) :
    _id(id)
{

}

MeshBound::~MeshBound()
{

}

glm::dvec3 MeshBound::operator()(const glm::dvec3& pos) const
{
    return pos;
}


const MeshBound MeshTopo::NO_BOUNDARY = MeshBound(0);

MeshTopo::MeshTopo() :
    isFixed(false),
    isBoundary(false),
    boundaryCallback(NO_BOUNDARY)
{
}

MeshTopo::MeshTopo(
        bool isFixed) :
    isFixed(isFixed),
    isBoundary(false),
    boundaryCallback(NO_BOUNDARY)
{
}

MeshTopo::MeshTopo(
        const MeshBound& boundaryCallback) :
    isFixed(false),
    isBoundary(&boundaryCallback != &NO_BOUNDARY),
    boundaryCallback(boundaryCallback)
{
}

void Mesh::clear()
{
    vert.clear();
    tetra.clear();
    prism.clear();
    hexa.clear();

    topo.clear();
}

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

double Mesh::hexahedronQuality(const MeshHex& hex)
{
    // Hexahedron quality ~= mean of two possible internal tetrahedrons
    MeshTet tetA(hex.v[0], hex.v[3], hex.v[5], hex.v[6]);
    MeshTet tetB(hex.v[1], hex.v[2], hex.v[7], hex.v[4]);
    double tetAQuality = tetrahedronQuality(tetA);
    double tetBQuality = tetrahedronQuality(tetB);
    return (tetAQuality + tetBQuality) / 2.0;
}


double Mesh::prismQuality(const MeshPri& pri)
{
    // Prism quality ~= mean of 6 possible tetrahedrons from prism triangular faces
    MeshTet tetA(pri.v[4], pri.v[1], pri.v[5], pri.v[3]);
    MeshTet tetB(pri.v[5], pri.v[2], pri.v[4], pri.v[0]);
    MeshTet tetC(pri.v[2], pri.v[1], pri.v[5], pri.v[3]);
    MeshTet tetD(pri.v[3], pri.v[2], pri.v[4], pri.v[0]);
    MeshTet tetE(pri.v[0], pri.v[1], pri.v[5], pri.v[3]);
    MeshTet tetF(pri.v[1], pri.v[2], pri.v[4], pri.v[0]);

    double tetAq = tetrahedronQuality(tetA);
    double tetBq = tetrahedronQuality(tetB);
    double tetCq = tetrahedronQuality(tetC);
    double tetDq = tetrahedronQuality(tetD);
    double tetEq = tetrahedronQuality(tetE);
    double tetFq = tetrahedronQuality(tetF);
    return (tetAq + tetBq + tetCq + tetDq + tetEq + tetFq) / (6.0 * 0.716178);
}

void Mesh::compileElementQuality(
        double& qualityMean,
        double& qualityVar,
        double& minQuality)
{
    int tetCount = tetra.size();
    int priCount = prism.size();
    int hexCount = hexa.size();

    int elemCount = tetCount + priCount + hexCount;
    std::vector<double> qualities(elemCount);
    int idx = 0;

    for(int i=0; i < tetCount; ++i, ++idx)
        qualities[idx] = tetrahedronQuality(tetra[i]);

    for(int i=0; i < priCount; ++i, ++idx)
        qualities[idx] = prismQuality(prism[i]);

    for(int i=0; i < hexCount; ++i, ++idx)
        qualities[idx] = hexahedronQuality(hexa[i]);


    minQuality = 1.0;
    qualityMean = 0.0;
    for(int i=0; i < elemCount; ++i)
    {
        if(qualities[i] < minQuality)
            minQuality = qualities[i];

        qualityMean = (qualityMean * i + qualities[i]) / (i + 1);
    }

    qualityVar = 0.0;
    for(int i=0; i < elemCount; ++i)
    {
        double qualityMeanDist = qualityMean - qualities[i];
        double qualityMeanDist2 = qualityMeanDist*qualityMeanDist;
        qualityVar = (qualityVar * 1 + qualityMeanDist2) / (1 + 1);
    }
}

void Mesh::compileVertexAdjacency()
{
    int vertCount = vert.size();

    topo.resize(vertCount);
    topo.shrink_to_fit();

    int tetCount = tetra.size();
    for(int i=0; i < tetCount; ++i)
    {
        for(int e=0; e < MeshTet::EDGE_COUNT; ++e)
        {
            addEdge(tetra[i].v[MeshTet::edges[e][0]],
                    tetra[i].v[MeshTet::edges[e][1]]);
        }
    }

    int prismCount = prism.size();
    for(int i=0; i < prismCount; ++i)
    {
        for(int e=0; e < MeshPri::EDGE_COUNT; ++e)
        {
            addEdge(prism[i].v[MeshPri::edges[e][0]],
                    prism[i].v[MeshPri::edges[e][1]]);
        }
    }

    int hexCount = hexa.size();
    for(int i=0; i < hexCount; ++i)
    {
        for(int e=0; e < MeshHex::EDGE_COUNT; ++e)
        {
            addEdge(hexa[i].v[MeshHex::edges[e][0]],
                    hexa[i].v[MeshHex::edges[e][1]]);
        }
    }

    for(int i=0; i < vertCount; ++i)
    {
        topo[i].neighbors.shrink_to_fit();
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
                         normal, false, quality);
        }
    }


    // Prisms
    int priCount = prism.size();
    for(int i=0; i < priCount; ++i)
    {
        const MeshPri& pri = prism[i];

        glm::dvec3 verts[] = {
            glm::dvec3(vert[pri[0]]),
            glm::dvec3(vert[pri[1]]),
            glm::dvec3(vert[pri[2]]),
            glm::dvec3(vert[pri[3]]),
            glm::dvec3(vert[pri[4]]),
            glm::dvec3(vert[pri[5]])
        };

        if(glm::dot(verts[0], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[1], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[2], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[3], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[4], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[5], cutNormal) - cutDistance > 0.0)
            continue;


        double quality = prismQuality(pri);

        for(int f=0; f < MeshPri::FACE_COUNT; ++f)
        {
            const MeshTri& tri = MeshPri::faces[f];
            glm::dvec3 A = verts[tri[1]] - verts[tri[0]];
            glm::dvec3 B = verts[tri[2]] - verts[tri[1]];
            glm::dvec3 normal = glm::normalize(glm::cross(A, B));
            pushTriangle(vertices, normals, triEdges, qualities,
                         verts[tri[0]], verts[tri[1]], verts[tri[2]],
                         normal, f < 7, quality);
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
                         normal, true, quality);
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
        bool fromQuad,
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

    if(fromQuad)
    {
        triEdges.push_back(255);
        triEdges.push_back(0);
        triEdges.push_back(0);

        triEdges.push_back(0);
        triEdges.push_back(255);
        triEdges.push_back(0);

        triEdges.push_back(255);
        triEdges.push_back(255);
        triEdges.push_back(0);
    }
    else
    {
        triEdges.push_back(0);
        triEdges.push_back(255);
        triEdges.push_back(255);

        triEdges.push_back(255);
        triEdges.push_back(0);
        triEdges.push_back(255);

        triEdges.push_back(255);
        triEdges.push_back(255);
        triEdges.push_back(0);
    }

    qualities.push_back(quality * 255);
    qualities.push_back(quality * 255);
    qualities.push_back(quality * 255);
}

void Mesh::addEdge(int firstVert, int secondVert)
{
    vector<int>& neighbors = topo[firstVert].neighbors;
    int neighborCount = neighbors.size();
    for(int n=0; n < neighborCount; ++n)
    {
        if(secondVert == neighbors[n])
            return;
    }

    // This really is a new edge
    topo[firstVert].neighbors.push_back(secondVert);
    topo[secondVert].neighbors.push_back(firstVert);
}
