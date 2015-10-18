#include "SolidAngleEvaluator.h"

using namespace glm;


SolidAngleEvaluator::SolidAngleEvaluator() :
    AbstractEvaluator(":/shaders/compute/Evaluating/SolidAngle.glsl")
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

    return determinant(dmat3(a, b, c)) /
            sqrt( 2.0 * (al*bl + dot(a, b)) *
                        (bl*cl + dot(b, c)) *
                        (cl*al + dot(c, a)));
}

double SolidAngleEvaluator::tetQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    double q0 = solidAngle(vp[1] - vp[0], vp[3] - vp[0], vp[2] - vp[0]);
    double q1 = solidAngle(vp[0] - vp[1], vp[2] - vp[1], vp[3] - vp[1]);
    double q2 = solidAngle(vp[0] - vp[2], vp[3] - vp[2], vp[1] - vp[2]);
    double q3 = solidAngle(vp[0] - vp[3], vp[1] - vp[3], vp[2] - vp[3]);

    double minQ = min(min(q0, q1),
                      min(q2, q3));

    return minQ * 3.67423461417; // 9 / sqrt(6)
}

double SolidAngleEvaluator::priQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    double q0 = solidAngle(vp[1] - vp[0], vp[2] - vp[0], vp[4] - vp[0]);
    double q1 = solidAngle(vp[0] - vp[1], vp[5] - vp[1], vp[3] - vp[1]);
    double q2 = solidAngle(vp[0] - vp[2], vp[3] - vp[2], vp[4] - vp[2]);
    double q3 = solidAngle(vp[1] - vp[3], vp[5] - vp[3], vp[2] - vp[3]);
    double q4 = solidAngle(vp[0] - vp[4], vp[2] - vp[4], vp[5] - vp[4]);
    double q5 = solidAngle(vp[1] - vp[5], vp[4] - vp[5], vp[3] - vp[5]);

    double minQ = min(min(q0, q1),
                      min(min(q2, q3),
                          min(q4, q5)));

    return minQ * 2.0; // 1.0 / <max val for regular prism>
}

double SolidAngleEvaluator::hexQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    double q0 = solidAngle(vp[1] - vp[0], vp[2] - vp[0], vp[4] - vp[0]);
    double q1 = solidAngle(vp[0] - vp[5], vp[3] - vp[1], vp[1] - vp[1]);
    double q2 = solidAngle(vp[0] - vp[2], vp[3] - vp[2], vp[6] - vp[2]);
    double q3 = solidAngle(vp[1] - vp[7], vp[2] - vp[3], vp[3] - vp[3]);
    double q4 = solidAngle(vp[0] - vp[6], vp[5] - vp[4], vp[4] - vp[4]);
    double q5 = solidAngle(vp[1] - vp[5], vp[4] - vp[5], vp[7] - vp[5]);
    double q6 = solidAngle(vp[2] - vp[6], vp[7] - vp[6], vp[4] - vp[6]);
    double q7 = solidAngle(vp[3] - vp[7], vp[5] - vp[7], vp[6] - vp[7]);

    double minQ = min(min(min(q0, q1), min(q2, q3)),
                      min(min(q4, q5), min(q6, q7)));

    return minQ * 1.41421356237; // sqrt(2)
}
