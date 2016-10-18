#include "MetricConformityEvaluator.h"

#include "Measurers/AbstractMeasurer.h"

using namespace glm;


const dmat3 TWO_I = dmat3(2.0);


// CUDA Drivers Interface
void installCudaMetricConformityEvaluator();


MetricConformityEvaluator::MetricConformityEvaluator() :
    AbstractEvaluator(":/glsl/compute/Evaluating/MetricConformity.glsl", installCudaMetricConformityEvaluator)
{
}

MetricConformityEvaluator::~MetricConformityEvaluator()
{

}

Metric MetricConformityEvaluator::specifiedMetric(
        const AbstractSampler& sampler,
        const dvec3& v0,
        const dvec3& v1,
        const dvec3& v2,
        const dvec3& v3,
        uint& cachedRefTet) const
{
    // Refs :
    //  P Keast, Moderate degree tetrahedral quadrature formulas, CMAME 55: 339-348 (1986)
    //  O. C. Zienkiewicz, The Finite Element Method,  Sixth Edition,
    // Taken from : http://www.cfd-online.com/Wiki/Code:_Quadrature_on_Tetrahedra

    const double H = 0.5;
    const double Q = (1.0 - H) / 3.0;
    return (sampler.metricAt((v0 + v1 + v2 + v3)/4.0,   cachedRefTet) * (-0.8) +
            sampler.metricAt(v0*H + v1*Q + v2*Q + v3*Q, cachedRefTet) * 0.45 +
            sampler.metricAt(v0*Q + v1*H + v2*Q + v3*Q, cachedRefTet) * 0.45 +
            sampler.metricAt(v0*Q + v1*Q + v2*H + v3*Q, cachedRefTet) * 0.45 +
            sampler.metricAt(v0*Q + v1*Q + v2*Q + v3*H, cachedRefTet) * 0.45);
}

double MetricConformityEvaluator::metricConformity(
        const dmat3& Fk,
        const Metric& Ms) const
{
    dmat3 Fk_inv = inverse(transpose(Fk));
    dmat3 Mk = Fk_inv * transpose(Fk_inv);
    dmat3 Mk_inv = inverse(Mk);
    dmat3 Ms_inv = inverse(Ms);

    dmat3 tNc = Mk_inv*Ms + Ms_inv*Mk - TWO_I;

    double tNc_frobenius2 =
            glm::dot(tNc[0], tNc[0]) +
            glm::dot(tNc[1], tNc[1]) +
            glm::dot(tNc[2], tNc[2]);

    if(!std::isnan(tNc_frobenius2))
    {
        double Fk_sign = sign(determinant(Fk));
        return Fk_sign / sqrt(1.0 + sqrt(tNc_frobenius2));
    }
    else
    {
        return 0.0;
    }
}

double MetricConformityEvaluator::tetQuality(
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[],
        const MeshTet& tet) const
{
    glm::dvec3 e03 = vp[3] - vp[0];
    glm::dvec3 e13 = vp[3] - vp[1];
    glm::dvec3 e23 = vp[3] - vp[2];

    glm::dmat3 Fk = dmat3(e03, e13, e23) * Fr_TET_INV;

    Metric Ms0 = specifiedMetric(sampler, vp[0], vp[1], vp[2], vp[3], tet.c[0]);

    double qual0 = metricConformity(Fk, Ms0);

    return qual0;
}

double MetricConformityEvaluator::priQuality(
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[],
        const MeshPri& pri) const
{
    dvec3 e03 = vp[3] - vp[0];
    dvec3 e14 = vp[4] - vp[1];
    dvec3 e25 = vp[5] - vp[2];
    dvec3 e01 = vp[1] - vp[0];
    dvec3 e12 = vp[2] - vp[1];
    dvec3 e20 = vp[0] - vp[2];
    dvec3 e34 = vp[4] - vp[3];
    dvec3 e45 = vp[5] - vp[4];
    dvec3 e53 = vp[3] - vp[5];

    // Prism corner quality is not invariant under edge swap
    // Third edge is the expected to be colinear with the first two's cross product
    dmat3 Fk0 = dmat3(-e01, e20, e03) * Fr_PRI_INV;
    dmat3 Fk1 = dmat3(-e12, e01, e14) * Fr_PRI_INV;
    dmat3 Fk2 = dmat3(-e20, e12, e25) * Fr_PRI_INV;
    dmat3 Fk3 = dmat3(-e34, e53, e03) * Fr_PRI_INV;
    dmat3 Fk4 = dmat3(-e45, e34, e14) * Fr_PRI_INV;
    dmat3 Fk5 = dmat3(-e53, e45, e25) * Fr_PRI_INV;

    Metric Ms0 = specifiedMetric(sampler, vp[0], vp[1], vp[2], vp[3], pri.c[0]);
    Metric Ms1 = specifiedMetric(sampler, vp[0], vp[1], vp[2], vp[4], pri.c[1]);
    Metric Ms2 = specifiedMetric(sampler, vp[0], vp[1], vp[2], vp[5], pri.c[2]);
    Metric Ms3 = specifiedMetric(sampler, vp[0], vp[3], vp[4], vp[5], pri.c[3]);
    Metric Ms4 = specifiedMetric(sampler, vp[1], vp[3], vp[4], vp[5], pri.c[4]);
    Metric Ms5 = specifiedMetric(sampler, vp[2], vp[3], vp[4], vp[5], pri.c[5]);

    double qual0 = metricConformity(Fk0, Ms0);
    double qual1 = metricConformity(Fk1, Ms1);
    double qual2 = metricConformity(Fk2, Ms2);
    double qual3 = metricConformity(Fk3, Ms3);
    double qual4 = metricConformity(Fk4, Ms4);
    double qual5 = metricConformity(Fk5, Ms5);


    if(qual0 <= 0.0 || qual1 <= 0.0 || qual2 <= 0.0 ||
       qual3 <= 0.0 || qual4 <= 0.0 || qual5 <= 0.0)
    {
        double qualMin = qual0;
        if(qual1 < qualMin) qualMin = qual1;
        if(qual2 < qualMin) qualMin = qual2;
        if(qual3 < qualMin) qualMin = qual3;
        if(qual4 < qualMin) qualMin = qual4;
        if(qual5 < qualMin) qualMin = qual5;
        return qualMin;
    }
    else
    {
        double mean = 6.0 / (
            1/qual0 + 1/qual1 + 1/qual2 +
            1/qual3 + 1/qual4 + 1/qual5);

        return mean;
    }
}

double MetricConformityEvaluator::hexQuality(
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const glm::dvec3 vp[],
        const MeshHex& hex) const
{
    dvec3 e01 = vp[1] - vp[0];
    dvec3 e03 = vp[3] - vp[0];
    dvec3 e04 = vp[4] - vp[0];
    dvec3 e12 = vp[2] - vp[1];
    dvec3 e15 = vp[5] - vp[1];
    dvec3 e23 = vp[3] - vp[2];
    dvec3 e26 = vp[6] - vp[2];
    dvec3 e37 = vp[7] - vp[3];
    dvec3 e45 = vp[5] - vp[4];
    dvec3 e47 = vp[7] - vp[4];
    dvec3 e56 = vp[6] - vp[5];
    dvec3 e67 = vp[7] - vp[6];

    // Since hex's corner matrix is the identity matrix,
    // there's no need to define Fr_INV.
    dmat3 Fk0 = dmat3(e01,  e04, -e03);
    dmat3 Fk1 = dmat3(e01,  e12,  e15);
    dmat3 Fk2 = dmat3(e12,  e26, -e23);
    dmat3 Fk3 = dmat3(e03,  e23,  e37);
    dmat3 Fk4 = dmat3(e04,  e45,  e47);
    dmat3 Fk5 = dmat3(e15, -e56,  e45);
    dmat3 Fk6 = dmat3(e26,  e56,  e67);
    dmat3 Fk7 = dmat3(e37,  e67, -e47);

    Metric Ms0 = specifiedMetric(sampler, vp[0], vp[1], vp[3], vp[4], hex.c[0]);
    Metric Ms1 = specifiedMetric(sampler, vp[0], vp[1], vp[2], vp[5], hex.c[1]);
    Metric Ms2 = specifiedMetric(sampler, vp[1], vp[2], vp[3], vp[6], hex.c[2]);
    Metric Ms3 = specifiedMetric(sampler, vp[0], vp[2], vp[3], vp[7], hex.c[3]);
    Metric Ms4 = specifiedMetric(sampler, vp[0], vp[4], vp[5], vp[7], hex.c[4]);
    Metric Ms5 = specifiedMetric(sampler, vp[1], vp[4], vp[5], vp[6], hex.c[5]);
    Metric Ms6 = specifiedMetric(sampler, vp[2], vp[5], vp[6], vp[7], hex.c[6]);
    Metric Ms7 = specifiedMetric(sampler, vp[3], vp[4], vp[6], vp[7], hex.c[7]);

    double qual0 = metricConformity(Fk0, Ms0);
    double qual1 = metricConformity(Fk1, Ms1);
    double qual2 = metricConformity(Fk2, Ms2);
    double qual3 = metricConformity(Fk3, Ms3);
    double qual4 = metricConformity(Fk4, Ms4);
    double qual5 = metricConformity(Fk5, Ms5);
    double qual6 = metricConformity(Fk6, Ms6);
    double qual7 = metricConformity(Fk7, Ms7);


    if(qual0 <= 0.0 || qual1 <= 0.0 || qual2 <= 0.0 || qual3 <= 0.0 ||
       qual4 <= 0.0 || qual5 <= 0.0 || qual6 <= 0.0 || qual7 <= 0.0)
    {
        double minQual = qual0;
        if(qual1 < minQual) minQual = qual1;
        if(qual2 < minQual) minQual = qual2;
        if(qual3 < minQual) minQual = qual3;
        if(qual4 < minQual) minQual = qual4;
        if(qual5 < minQual) minQual = qual5;
        if(qual6 < minQual) minQual = qual6;
        if(qual7 < minQual) minQual = qual7;
        return minQual;
    }
    else
    {
        double mean = 8.0 / (
            1/qual0 + 1/qual1 + 1/qual2 + 1/qual3 +
            1/qual4 + 1/qual5 + 1/qual6 + 1/qual7);

        return mean;
    }
}
