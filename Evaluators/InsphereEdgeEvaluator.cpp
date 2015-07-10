#include "InsphereEdgeEvaluator.h"


InsphereEdgeEvaluator::InsphereEdgeEvaluator() :
    AbstractEvaluator(":/shaders/compute/Quality/InsphereEdge.glsl")
{

}

InsphereEdgeEvaluator::~InsphereEdgeEvaluator()
{

}

double InsphereEdgeEvaluator::tetrahedronQuality(
        const Mesh& mesh, const MeshTet& tet) const
{
    glm::dvec3 A(mesh.vert[tet[0]]);
    glm::dvec3 B(mesh.vert[tet[1]]);
    glm::dvec3 C(mesh.vert[tet[2]]);
    glm::dvec3 D(mesh.vert[tet[3]]);

    double u = glm::distance(A, B);
    double v = glm::distance(A, C);
    double w = glm::distance(A, D);
    double U = glm::distance(C, D);
    double V = glm::distance(D, B);
    double W = glm::distance(B, C);

    double Volume = 4.0*u*u*v*v*w*w;
    Volume -= u*u*pow(v*v+w*w-U*U, 2.0);
    Volume -= v*v*pow(w*w+u*u-V*V, 2.0);
    Volume -= w*w*pow(u*u+v*v-W*W, 2.0);
    Volume += (v*v+w*w-U*U)*(w*w+u*u-V*V)*(u*u+v*v-W*W);
    Volume = sqrt(Volume);
    Volume /= 12.0;

    double s1 = (U + V + W) * 0.5;
    double s2 = (u + v + W) * 0.5;
    double s3 = (u + V + w) * 0.5;
    double s4 = (U + v + w) * 0.5;

    double L1 = sqrt(s1*(s1-U)*(s1-V)*(s1-W));
    double L2 = sqrt(s2*(s2-u)*(s2-v)*(s2-W));
    double L3 = sqrt(s3*(s3-u)*(s3-V)*(s3-w));
    double L4 = sqrt(s4*(s4-U)*(s4-v)*(s4-w));

    double R = (Volume*3)/(L1+L2+L3+L4);

    double maxLen = glm::max(glm::max(glm::max(u, v), w),
                             glm::max(glm::max(U, V), W));

    return (4.89897948557) * R / maxLen;
}

double InsphereEdgeEvaluator::prismQuality(
        const Mesh& mesh, const MeshPri& pri) const
{
    // Prism quality ~= mean of 6 possible tetrahedrons from prism triangular faces
    MeshTet tetA(pri.v[4], pri.v[1], pri.v[5], pri.v[3]);
    MeshTet tetB(pri.v[5], pri.v[2], pri.v[4], pri.v[0]);
    MeshTet tetC(pri.v[2], pri.v[1], pri.v[5], pri.v[3]);
    MeshTet tetD(pri.v[3], pri.v[2], pri.v[4], pri.v[0]);
    MeshTet tetE(pri.v[0], pri.v[1], pri.v[5], pri.v[3]);
    MeshTet tetF(pri.v[1], pri.v[2], pri.v[4], pri.v[0]);

    double tetAq = tetrahedronQuality(mesh, tetA);
    double tetBq = tetrahedronQuality(mesh, tetB);
    double tetCq = tetrahedronQuality(mesh, tetC);
    double tetDq = tetrahedronQuality(mesh, tetD);
    double tetEq = tetrahedronQuality(mesh, tetE);
    double tetFq = tetrahedronQuality(mesh, tetF);
    return (tetAq + tetBq + tetCq + tetDq + tetEq + tetFq)
            / 3.9067138981002011988; // C/6
}

double InsphereEdgeEvaluator::hexahedronQuality(
        const Mesh& mesh, const MeshHex& hex) const
{
    // Hexahedron quality ~= mean of two possible internal tetrahedrons
    MeshTet tetA(hex.v[0], hex.v[3], hex.v[5], hex.v[6]);
    MeshTet tetB(hex.v[1], hex.v[2], hex.v[7], hex.v[4]);
    double tetAQuality = tetrahedronQuality(mesh, tetA);
    double tetBQuality = tetrahedronQuality(mesh, tetB);
    return (tetAQuality + tetBQuality) * 0.5; // 1/2
}
