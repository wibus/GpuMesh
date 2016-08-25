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
        const vec3& v3,
        uint& cachedRefTet)
{
    const float H = 0.5;
    const float Q = (1.0 - H) / 3.0;
    return ((*metricAt)((v0 + v1 + v2 + v3)/4.0f,  cachedRefTet) * (-0.8f) +
            (*metricAt)(v0*H + v1*Q + v2*Q + v3*Q, cachedRefTet) * 0.45f +
            (*metricAt)(v0*Q + v1*H + v2*Q + v3*Q, cachedRefTet) * 0.45f +
            (*metricAt)(v0*Q + v1*Q + v2*H + v3*Q, cachedRefTet) * 0.45f +
            (*metricAt)(v0*Q + v1*Q + v2*Q + v3*H, cachedRefTet) * 0.45f);
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
    return Fk_sign / sqrt(1.0 + sqrt(tNc_frobenius2));
}

__device__ float metricConformityTetQuality(const vec3 vp[TET_VERTEX_COUNT], const Tet& tet)
{
    vec3 e03 = vp[3] - vp[0];
    vec3 e13 = vp[3] - vp[1];
    vec3 e23 = vp[3] - vp[2];

    mat3 Fk = mat3(e03, e13, e23) * Fr_TET_INV;

    mat3 Ms0 = specifiedMetric(vp[0], vp[1], vp[2], vp[3], tet.c[0]);

    float qual0 = metricConformity(Fk, Ms0);

    return qual0;
}

__device__ float metricConformityPriQuality(const vec3 vp[PRI_VERTEX_COUNT], const Pri& pri)
{
    vec3 e03 = vp[3] - vp[0];
    vec3 e14 = vp[4] - vp[1];
    vec3 e25 = vp[5] - vp[2];
    vec3 e01 = vp[1] - vp[0];
    vec3 e12 = vp[2] - vp[1];
    vec3 e20 = vp[0] - vp[2];
    vec3 e34 = vp[4] - vp[3];
    vec3 e45 = vp[5] - vp[4];
    vec3 e53 = vp[3] - vp[5];

    // Prism corner quality is not invariant under edge swap
    // Third edge is the expected to be colinear with the first two's cross product
    mat3 Fk0 = mat3(-e01, e20, e03) * Fr_PRI_INV;
    mat3 Fk1 = mat3(-e12, e01, e14) * Fr_PRI_INV;
    mat3 Fk2 = mat3(-e20, e12, e25) * Fr_PRI_INV;
    mat3 Fk3 = mat3(-e34, e53, e03) * Fr_PRI_INV;
    mat3 Fk4 = mat3(-e45, e34, e14) * Fr_PRI_INV;
    mat3 Fk5 = mat3(-e53, e45, e25) * Fr_PRI_INV;

    mat3 Ms0 = specifiedMetric(vp[0], vp[1], vp[2], vp[3], pri.c[0]);
    mat3 Ms1 = specifiedMetric(vp[0], vp[1], vp[2], vp[4], pri.c[1]);
    mat3 Ms2 = specifiedMetric(vp[0], vp[1], vp[2], vp[5], pri.c[2]);
    mat3 Ms3 = specifiedMetric(vp[0], vp[3], vp[4], vp[5], pri.c[3]);
    mat3 Ms4 = specifiedMetric(vp[1], vp[3], vp[4], vp[5], pri.c[4]);
    mat3 Ms5 = specifiedMetric(vp[2], vp[3], vp[4], vp[5], pri.c[5]);

    float qual0 = metricConformity(Fk0, Ms0);
    float qual1 = metricConformity(Fk1, Ms1);
    float qual2 = metricConformity(Fk2, Ms2);
    float qual3 = metricConformity(Fk3, Ms3);
    float qual4 = metricConformity(Fk4, Ms4);
    float qual5 = metricConformity(Fk5, Ms5);


    if(qual0 <= 0.0 || qual1 <= 0.0 || qual2 <= 0.0 ||
       qual3 <= 0.0 || qual4 <= 0.0 || qual5 <= 0.0)
    {
        float qualMin = qual0;
        if(qual1 < qualMin) qualMin = qual1;
        if(qual2 < qualMin) qualMin = qual2;
        if(qual3 < qualMin) qualMin = qual3;
        if(qual4 < qualMin) qualMin = qual4;
        if(qual5 < qualMin) qualMin = qual5;
        return qualMin;
    }
    else
    {
        float geomMean = exp( (-1.0 / 6.0) *
            (log(1.0/qual0) + log(1.0/qual1) + log(1.0/qual2) +
             log(1.0/qual3) + log(1.0/qual4) + log(1.0/qual5)));

        return geomMean;
    }
}

__device__ float metricConformityHexQuality(const vec3 vp[HEX_VERTEX_COUNT], const Hex& hex)
{
    vec3 e01 = vp[1] - vp[0];
    vec3 e03 = vp[3] - vp[0];
    vec3 e04 = vp[4] - vp[0];
    vec3 e12 = vp[2] - vp[1];
    vec3 e15 = vp[5] - vp[1];
    vec3 e23 = vp[3] - vp[2];
    vec3 e26 = vp[6] - vp[2];
    vec3 e37 = vp[7] - vp[3];
    vec3 e45 = vp[5] - vp[4];
    vec3 e47 = vp[7] - vp[4];
    vec3 e56 = vp[6] - vp[5];
    vec3 e67 = vp[7] - vp[6];

    // Since hex's corner matrix is the identity matrix,
    // there's no need to define Fr_INV.
    mat3 Fk0 = mat3(e01,  e04, -e03);
    mat3 Fk1 = mat3(e01,  e12,  e15);
    mat3 Fk2 = mat3(e12,  e26, -e23);
    mat3 Fk3 = mat3(e03,  e23,  e37);
    mat3 Fk4 = mat3(e04,  e45,  e47);
    mat3 Fk5 = mat3(e15, -e56,  e45);
    mat3 Fk6 = mat3(e26,  e56,  e67);
    mat3 Fk7 = mat3(e37,  e67, -e47);

    mat3 Ms0 = specifiedMetric(vp[0], vp[1], vp[3], vp[4], hex.c[0]);
    mat3 Ms1 = specifiedMetric(vp[0], vp[1], vp[2], vp[5], hex.c[1]);
    mat3 Ms2 = specifiedMetric(vp[1], vp[2], vp[3], vp[6], hex.c[2]);
    mat3 Ms3 = specifiedMetric(vp[0], vp[2], vp[3], vp[7], hex.c[3]);
    mat3 Ms4 = specifiedMetric(vp[0], vp[4], vp[5], vp[7], hex.c[4]);
    mat3 Ms5 = specifiedMetric(vp[1], vp[4], vp[5], vp[6], hex.c[5]);
    mat3 Ms6 = specifiedMetric(vp[2], vp[5], vp[6], vp[7], hex.c[6]);
    mat3 Ms7 = specifiedMetric(vp[3], vp[4], vp[6], vp[7], hex.c[7]);

    float qual0 = metricConformity(Fk0, Ms0);
    float qual1 = metricConformity(Fk1, Ms1);
    float qual2 = metricConformity(Fk2, Ms2);
    float qual3 = metricConformity(Fk3, Ms3);
    float qual4 = metricConformity(Fk4, Ms4);
    float qual5 = metricConformity(Fk5, Ms5);
    float qual6 = metricConformity(Fk6, Ms6);
    float qual7 = metricConformity(Fk7, Ms7);


    if(qual0 <= 0.0 || qual1 <= 0.0 || qual2 <= 0.0 || qual3 <= 0.0 ||
       qual4 <= 0.0 || qual5 <= 0.0 || qual6 <= 0.0 || qual7 <= 0.0)
    {
        float minQual = qual0;
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
        float geomMean = exp( (-1.0 / 8.0) *
            (log(1.0/qual0) + log(1.0/qual1) + log(1.0/qual2) + log(1.0/qual3) +
             log(1.0/qual4) + log(1.0/qual5) + log(1.0/qual6) + log(1.0/qual7)));

        return geomMean;
    }
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


    if(verboseCuda)
        printf("I -> CUDA \tMetric Conformity Evaluator installed\n");

    cudaCheckErrors("CUDA error during Metric Conformity installation");
}
