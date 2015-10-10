#include "MeanRatioEvaluator.h"

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

double MeanRatioEvaluator::tetQuality(const dvec3 vp[]) const
{
    const dmat3 Fr_INV(
        dvec3(1, 0, 0),
        dvec3(-0.5773502691896257645091, 1.154700538379251529018, 0),
        dvec3(-0.4082482904638630163662, -0.4082482904638630163662, 1.224744871391589049099)
    );

    dmat3 Tk0 = dmat3(vp[0]-vp[1], vp[0]-vp[2], vp[0]-vp[3]);

    double qual0 = cornerQuality(Tk0 * Fr_INV);

    // Shape measure is independent of chosen corner
    return qual0;
}

double MeanRatioEvaluator::priQuality(const dvec3 vp[]) const
{
    const dmat3 Fr_INV(
        dvec3(1.0, 0.0, 0.0),
        dvec3(-0.5773502691896257645091, 1.154700538379251529018, 0.0),
        dvec3(0.0, 0.0, 1.0)
    );

    // Prism corner quality is not invariant under edge swap
    // Third edge is the expected to be colinear with the two firsts' cross product
    dmat3 Tk0(vp[0]-vp[4], vp[0]-vp[2], vp[0]-vp[1]);
    dmat3 Tk1(vp[1]-vp[3], vp[1]-vp[5], vp[1]-vp[0]);
    dmat3 Tk2(vp[2]-vp[0], vp[2]-vp[4], vp[2]-vp[3]);
    dmat3 Tk3(vp[3]-vp[5], vp[3]-vp[1], vp[3]-vp[2]);
    dmat3 Tk4(vp[4]-vp[2], vp[4]-vp[0], vp[4]-vp[5]);
    dmat3 Tk5(vp[5]-vp[1], vp[5]-vp[3], vp[5]-vp[4]);

    double qual0 = cornerQuality(Tk0 * Fr_INV);
    double qual1 = cornerQuality(Tk1 * Fr_INV);
    double qual2 = cornerQuality(Tk2 * Fr_INV);
    double qual3 = cornerQuality(Tk3 * Fr_INV);
    double qual4 = cornerQuality(Tk4 * Fr_INV);
    double qual5 = cornerQuality(Tk5 * Fr_INV);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5) / 6.0;
}

double MeanRatioEvaluator::hexQuality(const dvec3 vp[]) const
{
    // Since hex's corner matrix is the identity matrix,
    // there's no need to define Fr_INV.

    dmat3 Tk0(vp[0]-vp[1], vp[0]-vp[4], vp[0]-vp[2]);
    dmat3 Tk1(vp[1]-vp[0], vp[1]-vp[3], vp[1]-vp[5]);
    dmat3 Tk2(vp[2]-vp[0], vp[2]-vp[6], vp[2]-vp[3]);
    dmat3 Tk3(vp[3]-vp[1], vp[3]-vp[2], vp[3]-vp[7]);
    dmat3 Tk4(vp[4]-vp[0], vp[4]-vp[5], vp[4]-vp[6]);
    dmat3 Tk5(vp[5]-vp[1], vp[5]-vp[7], vp[5]-vp[4]);
    dmat3 Tk6(vp[6]-vp[2], vp[6]-vp[4], vp[6]-vp[7]);
    dmat3 Tk7(vp[7]-vp[3], vp[7]-vp[6], vp[7]-vp[5]);

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
