#include "SolidAngleEvaluator.h"

using namespace glm;


SolidAngleEvaluator::SolidAngleEvaluator() :
    AbstractEvaluator(":/shaders/compute/Quality/SolidAngle.glsl")
{

}

SolidAngleEvaluator::~SolidAngleEvaluator()
{

}

inline double SolidAngleEvaluator::solidAngle(
        const dvec3& a,
        const dvec3& b,
        const dvec3& c) const
{
    double al = length(a);
    double bl = length(b);
    double cl = length(c);

    return abs(determinant(dmat3(a, b, c))) /
            sqrt( 2.0 * (al*bl + dot(a, b)) *
                        (bl*cl + dot(b, c)) *
                        (cl*al + dot(c, a)));
}

double SolidAngleEvaluator::tetQuality(const dvec3 verts[]) const
{
    double q0 = solidAngle(verts[1] - verts[0], verts[2] - verts[0], verts[3] - verts[0]);
    double q1 = solidAngle(verts[0] - verts[1], verts[2] - verts[1], verts[3] - verts[1]);
    double q2 = solidAngle(verts[0] - verts[2], verts[1] - verts[2], verts[3] - verts[2]);
    double q3 = solidAngle(verts[0] - verts[3], verts[1] - verts[3], verts[2] - verts[3]);

    double minQ = min(min(q0, q1),
                      min(q2, q3));
    return minQ * 3.67423461417; // 9 / sqrt(6)
}

double SolidAngleEvaluator::priQuality(const dvec3 verts[]) const
{
    double q0 = solidAngle(verts[1] - verts[0], verts[2] - verts[0], verts[4] - verts[0]);
    double q1 = solidAngle(verts[0] - verts[1], verts[3] - verts[1], verts[5] - verts[1]);
    double q2 = solidAngle(verts[0] - verts[2], verts[3] - verts[2], verts[4] - verts[2]);
    double q3 = solidAngle(verts[1] - verts[3], verts[2] - verts[3], verts[5] - verts[3]);
    double q4 = solidAngle(verts[0] - verts[4], verts[2] - verts[4], verts[5] - verts[4]);
    double q5 = solidAngle(verts[1] - verts[5], verts[3] - verts[5], verts[4] - verts[5]);

    double minQ = min(min(q0, q1),
                      min(min(q2, q3),
                          min(q4, q5)));
    return minQ * 2.61312592975; // 1.0 / <max val for regular prism>
}

double SolidAngleEvaluator::hexQuality(const dvec3 verts[]) const
{
    double q0 = solidAngle(verts[1] - verts[0], verts[2] - verts[0], verts[4] - verts[0]);
    double q1 = solidAngle(verts[0] - verts[1], verts[3] - verts[1], verts[5] - verts[1]);
    double q2 = solidAngle(verts[0] - verts[2], verts[3] - verts[2], verts[6] - verts[2]);
    double q3 = solidAngle(verts[1] - verts[3], verts[2] - verts[3], verts[7] - verts[3]);
    double q4 = solidAngle(verts[0] - verts[4], verts[5] - verts[4], verts[6] - verts[4]);
    double q5 = solidAngle(verts[1] - verts[5], verts[4] - verts[5], verts[7] - verts[5]);
    double q6 = solidAngle(verts[2] - verts[6], verts[4] - verts[6], verts[7] - verts[6]);
    double q7 = solidAngle(verts[3] - verts[7], verts[5] - verts[7], verts[6] - verts[7]);

    double minQ = min(min(min(q0, q1), min(q2, q3)),
                      min(min(q4, q5), min(q6, q7)));
    return minQ * 1.41421356237; // sqrt(2)
}
