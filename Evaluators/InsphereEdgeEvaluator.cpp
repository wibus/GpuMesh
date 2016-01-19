#include "InsphereEdgeEvaluator.h"

using namespace glm;


InsphereEdgeEvaluator::InsphereEdgeEvaluator() :
    AbstractEvaluator(":/glsl/compute/Evaluating/InsphereEdge.glsl")
{

}

InsphereEdgeEvaluator::~InsphereEdgeEvaluator()
{

}

double InsphereEdgeEvaluator::tetQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    double u = distance(vp[0], vp[1]);
    double v = distance(vp[0], vp[2]);
    double w = distance(vp[0], vp[3]);
    double U = distance(vp[2], vp[3]);
    double V = distance(vp[3], vp[1]);
    double W = distance(vp[1], vp[2]);

    double Volume = 4.0*u*u*v*v*w*w;
    Volume -= u*u*pow(v*v+w*w-U*U, 2.0);
    Volume -= v*v*pow(w*w+u*u-V*V, 2.0);
    Volume -= w*w*pow(u*u+v*v-W*W, 2.0);
    Volume += (v*v+w*w-U*U)*(w*w+u*u-V*V)*(u*u+v*v-W*W);
    Volume = sign(Volume) * sqrt(abs(Volume));
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

double InsphereEdgeEvaluator::priQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    // Prism quality ~= mean of 6 possible tetrahedrons from prism triangular faces
    const dvec3 tetA[] = {vp[4], vp[1], vp[5], vp[3]};
    const dvec3 tetB[] = {vp[5], vp[2], vp[4], vp[0]};
    const dvec3 tetC[] = {vp[2], vp[1], vp[5], vp[3]};
    const dvec3 tetD[] = {vp[3], vp[2], vp[4], vp[0]};
    const dvec3 tetE[] = {vp[0], vp[1], vp[5], vp[3]};
    const dvec3 tetF[] = {vp[1], vp[2], vp[4], vp[0]};

    double tetAq = tetQuality(discretizer, measurer, tetA);
    double tetBq = tetQuality(discretizer, measurer, tetB);
    double tetCq = tetQuality(discretizer, measurer, tetC);
    double tetDq = tetQuality(discretizer, measurer, tetD);
    double tetEq = tetQuality(discretizer, measurer, tetE);
    double tetFq = tetQuality(discretizer, measurer, tetF);
    return (tetAq + tetBq + tetCq + tetDq + tetEq + tetFq)
                / 4.2970697433826288147;
}

double InsphereEdgeEvaluator::hexQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    // Hexahedron quality ~= mean of two possible internal tetrahedrons
    const dvec3 tetA[] = {vp[0], vp[3], vp[5], vp[6]};
    const dvec3 tetB[] = {vp[1], vp[2], vp[7], vp[4]};
    double tetAQuality = tetQuality(discretizer, measurer, tetA);
    double tetBQuality = tetQuality(discretizer, measurer, tetB);
    return (tetAQuality + tetBQuality)
                / 2.0;
}
