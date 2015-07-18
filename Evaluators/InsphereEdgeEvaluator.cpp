#include "InsphereEdgeEvaluator.h"

using namespace glm;


InsphereEdgeEvaluator::InsphereEdgeEvaluator() :
    AbstractEvaluator(":/shaders/compute/Quality/InsphereEdge.glsl")
{

}

InsphereEdgeEvaluator::~InsphereEdgeEvaluator()
{

}

double InsphereEdgeEvaluator::tetQuality(const dvec3 verts[]) const
{
    double u = distance(verts[0], verts[1]);
    double v = distance(verts[0], verts[2]);
    double w = distance(verts[0], verts[3]);
    double U = distance(verts[2], verts[3]);
    double V = distance(verts[3], verts[1]);
    double W = distance(verts[1], verts[2]);

    double Volume = 4.0*u*u*v*v*w*w;
    Volume -= u*u*pow(v*v+w*w-U*U, 2.0);
    Volume -= v*v*pow(w*w+u*u-V*V, 2.0);
    Volume -= w*w*pow(u*u+v*v-W*W, 2.0);
    Volume += (v*v+w*w-U*U)*(w*w+u*u-V*V)*(u*u+v*v-W*W);
    Volume = sqrt(max(Volume, 0.0));
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

    double maxLen = max(max(max(u, v), w),
                             max(max(U, V), W));

    return (4.89897948557) * R / maxLen;
}

double InsphereEdgeEvaluator::priQuality(const dvec3 verts[]) const
{
    // Prism quality ~= mean of 6 possible tetrahedrons from prism triangular faces
    const dvec3 tetA[] = {verts[4], verts[1], verts[5], verts[3]};
    const dvec3 tetB[] = {verts[5], verts[2], verts[4], verts[0]};
    const dvec3 tetC[] = {verts[2], verts[1], verts[5], verts[3]};
    const dvec3 tetD[] = {verts[3], verts[2], verts[4], verts[0]};
    const dvec3 tetE[] = {verts[0], verts[1], verts[5], verts[3]};
    const dvec3 tetF[] = {verts[1], verts[2], verts[4], verts[0]};

    double tetAq = tetQuality(tetA);
    double tetBq = tetQuality(tetB);
    double tetCq = tetQuality(tetC);
    double tetDq = tetQuality(tetD);
    double tetEq = tetQuality(tetE);
    double tetFq = tetQuality(tetF);
    return (tetAq + tetBq + tetCq + tetDq + tetEq + tetFq)
                / 3.9067138981002011988;
}

double InsphereEdgeEvaluator::hexQuality(const dvec3 verts[]) const
{
    // Hexahedron quality ~= mean of two possible internal tetrahedrons
    const dvec3 tetA[] = {verts[0], verts[3], verts[5], verts[6]};
    const dvec3 tetB[] = {verts[1], verts[2], verts[7], verts[4]};
    double tetAQuality = tetQuality(tetA);
    double tetBQuality = tetQuality(tetB);
    return (tetAQuality + tetBQuality)
                / 2.0;
}
