#include "Base.cuh"


#define TWO_I mat3(2.0)

#define Fr_TET_INV mat3( \
    vec3(1, 0, 0), \
    vec3(-0.5773502691896257645091, 1.154700538379251529018, 0), \
    vec3(-0.4082482904638630163662, -0.4082482904638630163662, 1.224744871391589049099))

#define Fr_PRI_INV mat3( \
    vec3(1.0, 0.0, 0.0), \
    vec3(-0.5773502691896257645091, 1.154700538379251529018, 0.0), \
    vec3(0.0, 0.0, 1.0))

__device__ mat3 specifiedMetric(
        const vec3& v0,
        const vec3& v1,
        const vec3& v2,
        const vec3& v3)
{
    const float H = 0.5;
    const float Q = (1.0 - H) / 3.0;
    return ((*metricAt)((v0 + v1 + v2 + v3)/4.0f) * (-0.8f) +
            (*metricAt)(v0*H + v1*Q + v2*Q + v3*Q) * 0.45f +
            (*metricAt)(v0*Q + v1*H + v2*Q + v3*Q) * 0.45f +
            (*metricAt)(v0*Q + v1*Q + v2*H + v3*Q) * 0.45f +
            (*metricAt)(v0*Q + v1*Q + v2*Q + v3*H) * 0.45f);
}

__device__ float metricConformity(const mat3& Fk, const mat3& Ms)
{
    mat3 Fk_inv = inverse(transpose(Fk));
    mat3 Mk = Fk_inv * transpose(Fk_inv);
    mat3 Mk_inv = inverse(Mk);
    mat3 Ms_inv = inverse(Ms);

    mat3 tNc = Mk_inv*Ms + Ms_inv*Mk - TWO_I;

    float tNc_frobenius2 =
            dot(tNc[0], tNc[0]) +
            dot(tNc[1], tNc[1]) +
            dot(tNc[2], tNc[2]);

    float Fk_sign = sign(determinant(Fk));
    return Fk_sign / (1.0 + sqrt(tNc_frobenius2));
}

__device__ float metricConformityTetQuality(const vec3 vp[4])
{
    vec3 e10 = vp[0] - vp[1];
    vec3 e20 = vp[0] - vp[2];
    vec3 e30 = vp[0] - vp[3];

    mat3 Fk = mat3(e10, e20, e30) * Fr_TET_INV;

    mat3 Ms0 = specifiedMetric(vp[0], vp[1], vp[2], vp[3]);

    float qual0 = metricConformity(Fk, Ms0);

    return qual0;
}

__device__ float metricConformityPriQuality(const vec3 vp[6])
{
    vec3 e01 = vp[1] - vp[0];
    vec3 e02 = vp[2] - vp[0];
    vec3 e04 = vp[4] - vp[0];
    vec3 e13 = vp[3] - vp[1];
    vec3 e15 = vp[5] - vp[1];
    vec3 e23 = vp[3] - vp[2];
    vec3 e42 = vp[2] - vp[4];
    vec3 e35 = vp[5] - vp[3];
    vec3 e45 = vp[5] - vp[4];

    // Prism corner quality is not invariant under edge swap
    // Third edge is the expected to be colinear with the first two's cross product
    mat3 Fk0 = mat3(e02,  e04,  e01) * Fr_PRI_INV;
    mat3 Fk1 = mat3(e13,  e15,  e01) * Fr_PRI_INV;
    mat3 Fk2 = mat3(e02,  e42, -e23) * Fr_PRI_INV;
    mat3 Fk3 = mat3(e35, -e13,  e23) * Fr_PRI_INV;
    mat3 Fk4 = mat3(e42, -e04, -e45) * Fr_PRI_INV;
    mat3 Fk5 = mat3(e15,  e35,  e45) * Fr_PRI_INV;

    mat3 Ms0 = specifiedMetric(vp[0], vp[2], vp[4], vp[1]);
    mat3 Ms1 = specifiedMetric(vp[0], vp[1], vp[3], vp[5]);
    mat3 Ms2 = specifiedMetric(vp[0], vp[2], vp[3], vp[4]);
    mat3 Ms3 = specifiedMetric(vp[1], vp[2], vp[3], vp[5]);
    mat3 Ms4 = specifiedMetric(vp[0], vp[2], vp[4], vp[5]);
    mat3 Ms5 = specifiedMetric(vp[1], vp[3], vp[4], vp[5]);

    float qual0 = metricConformity(Fk0, Ms0);
    float qual1 = metricConformity(Fk1, Ms1);
    float qual2 = metricConformity(Fk2, Ms2);
    float qual3 = metricConformity(Fk3, Ms3);
    float qual4 = metricConformity(Fk4, Ms4);
    float qual5 = metricConformity(Fk5, Ms5);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5) / 6.0;
}

__device__ float metricConformityHexQuality(const vec3 vp[8])
{
    vec3 e01 = vp[1] - vp[0];
    vec3 e02 = vp[2] - vp[0];
    vec3 e04 = vp[4] - vp[0];
    vec3 e13 = vp[3] - vp[1];
    vec3 e15 = vp[5] - vp[1];
    vec3 e23 = vp[3] - vp[2];
    vec3 e26 = vp[6] - vp[2];
    vec3 e37 = vp[7] - vp[3];
    vec3 e45 = vp[5] - vp[4];
    vec3 e46 = vp[6] - vp[4];
    vec3 e57 = vp[7] - vp[5];
    vec3 e67 = vp[7] - vp[6];

    // Since hex's corner matrix is the identity matrix,
    // there's no need to define Fr_INV.
    mat3 Fk0 = mat3(e01,  e04, -e02);
    mat3 Fk1 = mat3(e01,  e13,  e15);
    mat3 Fk2 = mat3(e02,  e26,  e23);
    mat3 Fk3 = mat3(e13,  e23, -e37);
    mat3 Fk4 = mat3(e04,  e45,  e46);
    mat3 Fk5 = mat3(e15, -e57,  e45);
    mat3 Fk6 = mat3(e26,  e46, -e67);
    mat3 Fk7 = mat3(e37,  e67,  e57);

    mat3 Ms0 = specifiedMetric(vp[0], vp[1], vp[2], vp[4]);
    mat3 Ms1 = specifiedMetric(vp[0], vp[1], vp[3], vp[5]);
    mat3 Ms2 = specifiedMetric(vp[0], vp[2], vp[3], vp[6]);
    mat3 Ms3 = specifiedMetric(vp[1], vp[2], vp[3], vp[7]);
    mat3 Ms4 = specifiedMetric(vp[0], vp[4], vp[5], vp[6]);
    mat3 Ms5 = specifiedMetric(vp[1], vp[4], vp[5], vp[7]);
    mat3 Ms6 = specifiedMetric(vp[2], vp[4], vp[6], vp[7]);
    mat3 Ms7 = specifiedMetric(vp[3], vp[5], vp[6], vp[7]);

    float qual0 = metricConformity(Fk0, Ms0);
    float qual1 = metricConformity(Fk1, Ms1);
    float qual2 = metricConformity(Fk2, Ms2);
    float qual3 = metricConformity(Fk3, Ms3);
    float qual4 = metricConformity(Fk4, Ms4);
    float qual5 = metricConformity(Fk5, Ms5);
    float qual6 = metricConformity(Fk6, Ms6);
    float qual7 = metricConformity(Fk7, Ms7);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5 + qual6 + qual7) / 8.0;
}

__device__ tetQualityFct metricConformityTetQualityPtr = metricConformityTetQuality;
__device__ priQualityFct metricConformityPriQualityPtr = metricConformityPriQuality;
__device__ hexQualityFct metricConformityHexQualityPtr = metricConformityHexQuality;


// CUDA Drivers
void installCudaMetricConformityEvaluator()
{
    tetQualityFct d_tetQuality = nullptr;
    cudaMemcpyFromSymbol(&d_tetQuality, metricConformityTetQualityPtr, sizeof(tetQualityFct));
    cudaMemcpyToSymbol(tetQualityImpl, &d_tetQuality, sizeof(tetQualityFct));

    priQualityFct d_priQuality = nullptr;
    cudaMemcpyFromSymbol(&d_priQuality, metricConformityPriQualityPtr, sizeof(priQualityFct));
    cudaMemcpyToSymbol(priQualityImpl, &d_priQuality, sizeof(priQualityFct));

    hexQualityFct d_hexQuality = nullptr;
    cudaMemcpyFromSymbol(&d_hexQuality, metricConformityHexQualityPtr, sizeof(hexQualityFct));
    cudaMemcpyToSymbol(hexQualityImpl, &d_hexQuality, sizeof(hexQualityFct));

    printf("I -> CUDA \tMetric Conformity Evaluator installed\n");
}
