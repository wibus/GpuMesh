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
        const AbstractDiscretizer& discretizer,
        const dvec3& v0,
        const dvec3& v1,
        const dvec3& v2,
        const dvec3& v3) const
{
    // Refs :
    //  P Keast, Moderate degree tetrahedral quadrature formulas, CMAME 55: 339-348 (1986)
    //  O. C. Zienkiewicz, The Finite Element Method,  Sixth Edition,
    // Taken from : http://www.cfd-online.com/Wiki/Code:_Quadrature_on_Tetrahedra

    const double H = 0.5;
    const double Q = (1.0 - H) / 3.0;
    return (discretizer.metricAt((v0 + v1 + v2 + v3)/4.0) * (-0.8) +
            discretizer.metricAt(v0*H + v1*Q + v2*Q + v3*Q) * 0.45 +
            discretizer.metricAt(v0*Q + v1*H + v2*Q + v3*Q) * 0.45 +
            discretizer.metricAt(v0*Q + v1*Q + v2*H + v3*Q) * 0.45 +
            discretizer.metricAt(v0*Q + v1*Q + v2*Q + v3*H) * 0.45);
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

    double Fk_sign = sign(determinant(Fk));
    return Fk_sign / (1.0 + sqrt(tNc_frobenius2));
}

double MetricConformityEvaluator::tetQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const dvec3 vp[]) const
{
    glm::dvec3 e10 = vp[0] - vp[1];
    glm::dvec3 e20 = vp[0] - vp[2];
    glm::dvec3 e30 = vp[0] - vp[3];

    glm::dmat3 Fk = dmat3(e10, e20, e30) * Fr_TET_INV;

    Metric Ms0 = specifiedMetric(discretizer, vp[0], vp[1], vp[2], vp[3]);

    double qual0 = metricConformity(Fk, Ms0);

    return qual0;
}

double MetricConformityEvaluator::priQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const dvec3 vp[]) const
{
    dvec3 e01 = vp[1] - vp[0];
    dvec3 e02 = vp[2] - vp[0];
    dvec3 e04 = vp[4] - vp[0];
    dvec3 e13 = vp[3] - vp[1];
    dvec3 e15 = vp[5] - vp[1];
    dvec3 e23 = vp[3] - vp[2];
    dvec3 e42 = vp[2] - vp[4];
    dvec3 e35 = vp[5] - vp[3];
    dvec3 e45 = vp[5] - vp[4];

    // Prism corner quality is not invariant under edge swap
    // Third edge is the expected to be colinear with the first two's cross product
    dmat3 Fk0 = dmat3(e02,  e04,  e01) * Fr_PRI_INV;
    dmat3 Fk1 = dmat3(e13,  e15,  e01) * Fr_PRI_INV;
    dmat3 Fk2 = dmat3(e02,  e42, -e23) * Fr_PRI_INV;
    dmat3 Fk3 = dmat3(e35, -e13,  e23) * Fr_PRI_INV;
    dmat3 Fk4 = dmat3(e42, -e04, -e45) * Fr_PRI_INV;
    dmat3 Fk5 = dmat3(e15,  e35,  e45) * Fr_PRI_INV;

    Metric Ms0 = specifiedMetric(discretizer, vp[0], vp[2], vp[4], vp[1]);
    Metric Ms1 = specifiedMetric(discretizer, vp[0], vp[1], vp[3], vp[5]);
    Metric Ms2 = specifiedMetric(discretizer, vp[0], vp[2], vp[3], vp[4]);
    Metric Ms3 = specifiedMetric(discretizer, vp[1], vp[2], vp[3], vp[5]);
    Metric Ms4 = specifiedMetric(discretizer, vp[0], vp[2], vp[4], vp[5]);
    Metric Ms5 = specifiedMetric(discretizer, vp[1], vp[3], vp[4], vp[5]);

    double qual0 = metricConformity(Fk0, Ms0);
    double qual1 = metricConformity(Fk1, Ms1);
    double qual2 = metricConformity(Fk2, Ms2);
    double qual3 = metricConformity(Fk3, Ms3);
    double qual4 = metricConformity(Fk4, Ms4);
    double qual5 = metricConformity(Fk5, Ms5);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5) / 6.0;
}

double MetricConformityEvaluator::hexQuality(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const dvec3 vp[]) const
{
    dvec3 e01 = vp[1] - vp[0];
    dvec3 e02 = vp[2] - vp[0];
    dvec3 e04 = vp[4] - vp[0];
    dvec3 e13 = vp[3] - vp[1];
    dvec3 e15 = vp[5] - vp[1];
    dvec3 e23 = vp[3] - vp[2];
    dvec3 e26 = vp[6] - vp[2];
    dvec3 e37 = vp[7] - vp[3];
    dvec3 e45 = vp[5] - vp[4];
    dvec3 e46 = vp[6] - vp[4];
    dvec3 e57 = vp[7] - vp[5];
    dvec3 e67 = vp[7] - vp[6];

    // Since hex's corner matrix is the identity matrix,
    // there's no need to define Fr_INV.
    dmat3 Fk0 = dmat3(e01,  e04, -e02);
    dmat3 Fk1 = dmat3(e01,  e13,  e15);
    dmat3 Fk2 = dmat3(e02,  e26,  e23);
    dmat3 Fk3 = dmat3(e13,  e23, -e37);
    dmat3 Fk4 = dmat3(e04,  e45,  e46);
    dmat3 Fk5 = dmat3(e15, -e57,  e45);
    dmat3 Fk6 = dmat3(e26,  e46, -e67);
    dmat3 Fk7 = dmat3(e37,  e67,  e57);

    Metric Ms0 = specifiedMetric(discretizer, vp[0], vp[1], vp[2], vp[4]);
    Metric Ms1 = specifiedMetric(discretizer, vp[0], vp[1], vp[3], vp[5]);
    Metric Ms2 = specifiedMetric(discretizer, vp[0], vp[2], vp[3], vp[6]);
    Metric Ms3 = specifiedMetric(discretizer, vp[1], vp[2], vp[3], vp[7]);
    Metric Ms4 = specifiedMetric(discretizer, vp[0], vp[4], vp[5], vp[6]);
    Metric Ms5 = specifiedMetric(discretizer, vp[1], vp[4], vp[5], vp[7]);
    Metric Ms6 = specifiedMetric(discretizer, vp[2], vp[4], vp[6], vp[7]);
    Metric Ms7 = specifiedMetric(discretizer, vp[3], vp[5], vp[6], vp[7]);

    double qual0 = metricConformity(Fk0, Ms0);
    double qual1 = metricConformity(Fk1, Ms1);
    double qual2 = metricConformity(Fk2, Ms2);
    double qual3 = metricConformity(Fk3, Ms3);
    double qual4 = metricConformity(Fk4, Ms4);
    double qual5 = metricConformity(Fk5, Ms5);
    double qual6 = metricConformity(Fk6, Ms6);
    double qual7 = metricConformity(Fk7, Ms7);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5 + qual6 + qual7) / 8.0;
}
