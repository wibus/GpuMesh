#include "Base.cuh"

// TODO wbussiere 2016-01-26 : Plug measurer function pointer
__device__ vec3 riemannianSegment(const vec3& a, const vec3& b)
{
    return b - a;
}


#define Fr_TET_INV mat3( \
    vec3(1, 0, 0), \
    vec3(-0.5773502691896257645091, 1.154700538379251529018, 0), \
    vec3(-0.4082482904638630163662, -0.4082482904638630163662, 1.224744871391589049099))

#define Fr_PRI_INV mat3( \
    vec3(1.0, 0.0, 0.0), \
    vec3(-0.5773502691896257645091, 1.154700538379251529018, 0.0), \
    vec3(0.0, 0.0, 1.0))

__device__ float cornerQuality(const mat3& Fk)
{
    float Fk_det = determinant(Fk);

    float Fk_frobenius2 =
        dot(Fk[0], Fk[0]) +
        dot(Fk[1], Fk[1]) +
        dot(Fk[2], Fk[2]);

    return sign(Fk_det) * 3.0 * pow(abs(Fk_det), 2.0/3.0) / Fk_frobenius2;
}

__device__ float meanRatioTetQuality(const vec3 vp[TET_VERTEX_COUNT])
{
    vec3 e10 = riemannianSegment(vp[1], vp[0]);
    vec3 e20 = riemannianSegment(vp[2], vp[0]);
    vec3 e30 = riemannianSegment(vp[3], vp[0]);

    mat3 Tk0 = mat3(e10, e20, e30);

    float qual0 = cornerQuality(Tk0 * Fr_TET_INV);

    // Shape measure is independent of chosen corner
    return qual0;
}

__device__ float meanRatioPriQuality(const vec3 vp[PRI_VERTEX_COUNT])
{
    vec3 e01 = riemannianSegment(vp[0], vp[1]);
    vec3 e02 = riemannianSegment(vp[0], vp[2]);
    vec3 e04 = riemannianSegment(vp[0], vp[4]);
    vec3 e13 = riemannianSegment(vp[1], vp[3]);
    vec3 e15 = riemannianSegment(vp[1], vp[5]);
    vec3 e23 = riemannianSegment(vp[2], vp[3]);
    vec3 e42 = riemannianSegment(vp[4], vp[2]);
    vec3 e35 = riemannianSegment(vp[3], vp[5]);
    vec3 e45 = riemannianSegment(vp[4], vp[5]);

    // Prism corner quality is not invariant under edge swap
    // Third edge is the expected to be colinear with the first two cross product
    mat3 Tk0 = mat3(e02,  e04,  e01);
    mat3 Tk1 = mat3(e13,  e15,  e01);
    mat3 Tk2 = mat3(e02,  e42, -e23);
    mat3 Tk3 = mat3(e35, -e13,  e23);
    mat3 Tk4 = mat3(e42, -e04, -e45);
    mat3 Tk5 = mat3(e15,  e35,  e45);

    float qual0 = cornerQuality(Tk0 * Fr_PRI_INV);
    float qual1 = cornerQuality(Tk1 * Fr_PRI_INV);
    float qual2 = cornerQuality(Tk2 * Fr_PRI_INV);
    float qual3 = cornerQuality(Tk3 * Fr_PRI_INV);
    float qual4 = cornerQuality(Tk4 * Fr_PRI_INV);
    float qual5 = cornerQuality(Tk5 * Fr_PRI_INV);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5) / 6.0;
}

__device__ float meanRatioHexQuality(const vec3 vp[HEX_VERTEX_COUNT])
{
    // Since hex's corner matrix is the identity matrix,
    // there's no need to define Fr_INV.
    vec3 e01 = riemannianSegment(vp[0], vp[1]);
    vec3 e02 = riemannianSegment(vp[0], vp[2]);
    vec3 e04 = riemannianSegment(vp[0], vp[4]);
    vec3 e13 = riemannianSegment(vp[1], vp[3]);
    vec3 e15 = riemannianSegment(vp[1], vp[5]);
    vec3 e23 = riemannianSegment(vp[2], vp[3]);
    vec3 e26 = riemannianSegment(vp[2], vp[6]);
    vec3 e37 = riemannianSegment(vp[3], vp[7]);
    vec3 e45 = riemannianSegment(vp[4], vp[5]);
    vec3 e46 = riemannianSegment(vp[4], vp[6]);
    vec3 e57 = riemannianSegment(vp[5], vp[7]);
    vec3 e67 = riemannianSegment(vp[6], vp[7]);

    mat3 Tk0 = mat3(e01,  e04, -e02);
    mat3 Tk1 = mat3(e01,  e13,  e15);
    mat3 Tk2 = mat3(e02,  e26,  e23);
    mat3 Tk3 = mat3(e13,  e23, -e37);
    mat3 Tk4 = mat3(e04,  e45,  e46);
    mat3 Tk5 = mat3(e15, -e57,  e45);
    mat3 Tk6 = mat3(e26,  e46, -e67);
    mat3 Tk7 = mat3(e37,  e67,  e57);

    float qual0 = cornerQuality(Tk0);
    float qual1 = cornerQuality(Tk1);
    float qual2 = cornerQuality(Tk2);
    float qual3 = cornerQuality(Tk3);
    float qual4 = cornerQuality(Tk4);
    float qual5 = cornerQuality(Tk5);
    float qual6 = cornerQuality(Tk6);
    float qual7 = cornerQuality(Tk7);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5 + qual6 + qual7) / 8.0;
}

__device__ tetQualityFct meanRatioTetQualityPtr = meanRatioTetQuality;
__device__ priQualityFct meanRatioPriQualityPtr = meanRatioPriQuality;
__device__ hexQualityFct meanRatioHexQualityPtr = meanRatioHexQuality;


// CUDA Drivers
void installCudaMeanRatioEvaluator()
{
    tetQualityFct d_tetQuality = nullptr;
    cudaMemcpyFromSymbol(&d_tetQuality, meanRatioTetQualityPtr, sizeof(tetQualityFct));
    cudaMemcpyToSymbol(tetQualityImpl, &d_tetQuality, sizeof(tetQualityFct));

    priQualityFct d_priQuality = nullptr;
    cudaMemcpyFromSymbol(&d_priQuality, meanRatioPriQualityPtr, sizeof(priQualityFct));
    cudaMemcpyToSymbol(priQualityImpl, &d_priQuality, sizeof(priQualityFct));

    hexQualityFct d_hexQuality = nullptr;
    cudaMemcpyFromSymbol(&d_hexQuality, meanRatioHexQualityPtr, sizeof(hexQualityFct));
    cudaMemcpyToSymbol(hexQualityImpl, &d_hexQuality, sizeof(hexQualityFct));

    printf("I -> CUDA \tMean Ration Evaluator installed\n");
}
