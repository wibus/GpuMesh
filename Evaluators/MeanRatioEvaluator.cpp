#include "MeanRatioEvaluator.h"

#include "Measurers/AbstractMeasurer.h"

using namespace glm;


// CUDA Drivers Interface
void installCudaMeanRatioEvaluator();


MeanRatioEvaluator::MeanRatioEvaluator() :
    AbstractEvaluator(":/glsl/compute/Evaluating/MeanRatio.glsl", installCudaMeanRatioEvaluator)
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
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const dvec3 vp[],
        const MeshTet& tet) const
{
    glm::dvec3 e03 = measurer.riemannianSegment(sampler, vp[0], vp[3], tet.c[0]);
    glm::dvec3 e13 = measurer.riemannianSegment(sampler, vp[1], vp[3], tet.c[0]);
    glm::dvec3 e23 = measurer.riemannianSegment(sampler, vp[2], vp[3], tet.c[0]);

    dmat3 Fk0 = dmat3(e03, e13, e23) * Fr_TET_INV;

    double qual0 = cornerQuality(Fk0);

    // Shape measure is independent of chosen corner
    return qual0;
}

double MeanRatioEvaluator::priQuality(
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[],
        const MeshPri& pri) const
{
    glm::dvec3 e03 = measurer.riemannianSegment(sampler, vp[0], vp[3], pri.c[0]);
    glm::dvec3 e14 = measurer.riemannianSegment(sampler, vp[1], vp[4], pri.c[1]);
    glm::dvec3 e25 = measurer.riemannianSegment(sampler, vp[2], vp[5], pri.c[2]);
    glm::dvec3 e01 = measurer.riemannianSegment(sampler, vp[0], vp[1], pri.c[0]);
    glm::dvec3 e12 = measurer.riemannianSegment(sampler, vp[1], vp[2], pri.c[1]);
    glm::dvec3 e20 = measurer.riemannianSegment(sampler, vp[2], vp[0], pri.c[2]);
    glm::dvec3 e34 = measurer.riemannianSegment(sampler, vp[3], vp[4], pri.c[3]);
    glm::dvec3 e45 = measurer.riemannianSegment(sampler, vp[4], vp[5], pri.c[4]);
    glm::dvec3 e53 = measurer.riemannianSegment(sampler, vp[5], vp[3], pri.c[5]);

    // Prism corner quality is not invariant under edge swap
    // Third edge is the expected to be colinear with the first two cross product
    dmat3 Fk0 = dmat3(-e01, e20, e03) * Fr_PRI_INV;
    dmat3 Fk1 = dmat3(-e12, e01, e14) * Fr_PRI_INV;
    dmat3 Fk2 = dmat3(-e20, e12, e25) * Fr_PRI_INV;
    dmat3 Fk3 = dmat3(-e34, e53, e03) * Fr_PRI_INV;
    dmat3 Fk4 = dmat3(-e45, e34, e14) * Fr_PRI_INV;
    dmat3 Fk5 = dmat3(-e53, e45, e25) * Fr_PRI_INV;

    double qual0 = cornerQuality(Fk0);
    double qual1 = cornerQuality(Fk1);
    double qual2 = cornerQuality(Fk2);
    double qual3 = cornerQuality(Fk3);
    double qual4 = cornerQuality(Fk4);
    double qual5 = cornerQuality(Fk5);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5) / 6.0;
}

double MeanRatioEvaluator::hexQuality(
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[],
        const MeshHex& hex) const
{
    // Since hex's corner matrix is the identity matrix,
    // there's no need to define Fr_HEX_INV.
    glm::dvec3 e01 = measurer.riemannianSegment(sampler, vp[0], vp[1], hex.c[0]);
    glm::dvec3 e03 = measurer.riemannianSegment(sampler, vp[0], vp[3], hex.c[0]);
    glm::dvec3 e04 = measurer.riemannianSegment(sampler, vp[0], vp[4], hex.c[0]);
    glm::dvec3 e12 = measurer.riemannianSegment(sampler, vp[1], vp[2], hex.c[1]);
    glm::dvec3 e15 = measurer.riemannianSegment(sampler, vp[1], vp[5], hex.c[1]);
    glm::dvec3 e23 = measurer.riemannianSegment(sampler, vp[2], vp[3], hex.c[3]);
    glm::dvec3 e26 = measurer.riemannianSegment(sampler, vp[2], vp[6], hex.c[2]);
    glm::dvec3 e37 = measurer.riemannianSegment(sampler, vp[3], vp[7], hex.c[3]);
    glm::dvec3 e45 = measurer.riemannianSegment(sampler, vp[4], vp[5], hex.c[4]);
    glm::dvec3 e47 = measurer.riemannianSegment(sampler, vp[4], vp[7], hex.c[4]);
    glm::dvec3 e56 = measurer.riemannianSegment(sampler, vp[5], vp[6], hex.c[5]);
    glm::dvec3 e67 = measurer.riemannianSegment(sampler, vp[6], vp[7], hex.c[6]);

    dmat3 Fk0 = dmat3(e01,  e04, -e03);
    dmat3 Fk1 = dmat3(e01,  e12,  e15);
    dmat3 Fk2 = dmat3(e12,  e26, -e23);
    dmat3 Fk3 = dmat3(e03,  e23,  e37);
    dmat3 Fk4 = dmat3(e04,  e45,  e47);
    dmat3 Fk5 = dmat3(e15, -e56,  e45);
    dmat3 Fk6 = dmat3(e26,  e56,  e67);
    dmat3 Fk7 = dmat3(e37,  e67, -e47);

    double qual0 = cornerQuality(Fk0);
    double qual1 = cornerQuality(Fk1);
    double qual2 = cornerQuality(Fk2);
    double qual3 = cornerQuality(Fk3);
    double qual4 = cornerQuality(Fk4);
    double qual5 = cornerQuality(Fk5);
    double qual6 = cornerQuality(Fk6);
    double qual7 = cornerQuality(Fk7);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5 + qual6 + qual7) / 8.0;
}
