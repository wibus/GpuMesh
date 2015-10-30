#include "MeanRatioEvaluator.h"

#include "Measurers/AbstractMeasurer.h"

using namespace glm;


MeanRatioEvaluator::MeanRatioEvaluator() :
    AbstractEvaluator(":/shaders/compute/Evaluating/MeanRatio.glsl")
{
}

MeanRatioEvaluator::~MeanRatioEvaluator()
{

}

double MeanRatioEvaluator::cornerQuality(const dmat3& Fk) const
{
    double Fk_det = determinant(Fk);

    double Fk_frobenius2 =
        dot(Fk[0], Fk[0]) +
        dot(Fk[1], Fk[1]) +
        dot(Fk[2], Fk[2]);

    return sign(Fk_det) * 3.0 * pow(abs(Fk_det), 2.0/3.0) / Fk_frobenius2;
}

double MeanRatioEvaluator::tetQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    glm::dvec3 e10 = measurer.riemannianSegment(discretizer, vp[1], vp[0]);
    glm::dvec3 e20 = measurer.riemannianSegment(discretizer, vp[2], vp[0]);
    glm::dvec3 e30 = measurer.riemannianSegment(discretizer, vp[3], vp[0]);

    dmat3 Tk0 = dmat3(e10, e20, e30);

    double qual0 = cornerQuality(Tk0 * Fr_TET_INV);

    // Shape measure is independent of chosen corner
    return qual0;
}

double MeanRatioEvaluator::priQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    glm::dvec3 e01 = measurer.riemannianSegment(discretizer, vp[0], vp[1]);
    glm::dvec3 e02 = measurer.riemannianSegment(discretizer, vp[0], vp[2]);
    glm::dvec3 e04 = measurer.riemannianSegment(discretizer, vp[0], vp[4]);
    glm::dvec3 e13 = measurer.riemannianSegment(discretizer, vp[1], vp[3]);
    glm::dvec3 e15 = measurer.riemannianSegment(discretizer, vp[1], vp[5]);
    glm::dvec3 e23 = measurer.riemannianSegment(discretizer, vp[2], vp[3]);
    glm::dvec3 e42 = measurer.riemannianSegment(discretizer, vp[4], vp[2]);
    glm::dvec3 e35 = measurer.riemannianSegment(discretizer, vp[3], vp[5]);
    glm::dvec3 e45 = measurer.riemannianSegment(discretizer, vp[4], vp[5]);

    // Prism corner quality is not invariant under edge swap
    // Third edge is the expected to be colinear with the first two cross product
    dmat3 Tk0 = dmat3(e02,  e04,  e01);
    dmat3 Tk1 = dmat3(e13,  e15,  e01);
    dmat3 Tk2 = dmat3(e02,  e42, -e23);
    dmat3 Tk3 = dmat3(e35, -e13,  e23);
    dmat3 Tk4 = dmat3(e42, -e04, -e45);
    dmat3 Tk5 = dmat3(e15,  e35,  e45);

    double qual0 = cornerQuality(Tk0 * Fr_PRI_INV);
    double qual1 = cornerQuality(Tk1 * Fr_PRI_INV);
    double qual2 = cornerQuality(Tk2 * Fr_PRI_INV);
    double qual3 = cornerQuality(Tk3 * Fr_PRI_INV);
    double qual4 = cornerQuality(Tk4 * Fr_PRI_INV);
    double qual5 = cornerQuality(Tk5 * Fr_PRI_INV);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5) / 6.0;
}

double MeanRatioEvaluator::hexQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[]) const
{
    // Since hex's corner matrix is the identity matrix,
    // there's no need to define Fr_INV.
    glm::dvec3 e01 = measurer.riemannianSegment(discretizer, vp[0], vp[1]);
    glm::dvec3 e02 = measurer.riemannianSegment(discretizer, vp[0], vp[2]);
    glm::dvec3 e04 = measurer.riemannianSegment(discretizer, vp[0], vp[4]);
    glm::dvec3 e13 = measurer.riemannianSegment(discretizer, vp[1], vp[3]);
    glm::dvec3 e15 = measurer.riemannianSegment(discretizer, vp[1], vp[5]);
    glm::dvec3 e23 = measurer.riemannianSegment(discretizer, vp[2], vp[3]);
    glm::dvec3 e26 = measurer.riemannianSegment(discretizer, vp[2], vp[6]);
    glm::dvec3 e37 = measurer.riemannianSegment(discretizer, vp[3], vp[7]);
    glm::dvec3 e45 = measurer.riemannianSegment(discretizer, vp[4], vp[5]);
    glm::dvec3 e46 = measurer.riemannianSegment(discretizer, vp[4], vp[6]);
    glm::dvec3 e57 = measurer.riemannianSegment(discretizer, vp[5], vp[7]);
    glm::dvec3 e67 = measurer.riemannianSegment(discretizer, vp[6], vp[7]);

    dmat3 Tk0 = dmat3(e01,  e04, -e02);
    dmat3 Tk1 = dmat3(e01,  e13,  e15);
    dmat3 Tk2 = dmat3(e02,  e26,  e23);
    dmat3 Tk3 = dmat3(e13,  e23, -e37);
    dmat3 Tk4 = dmat3(e04,  e45,  e46);
    dmat3 Tk5 = dmat3(e15, -e57,  e45);
    dmat3 Tk6 = dmat3(e26,  e46, -e67);
    dmat3 Tk7 = dmat3(e37,  e67,  e57);

    double qual0 = cornerQuality(Tk0);
    double qual1 = cornerQuality(Tk1);
    double qual2 = cornerQuality(Tk2);
    double qual3 = cornerQuality(Tk3);
    double qual4 = cornerQuality(Tk4);
    double qual5 = cornerQuality(Tk5);
    double qual6 = cornerQuality(Tk6);
    double qual7 = cornerQuality(Tk7);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5 + qual6 + qual7) / 8.0;
}
