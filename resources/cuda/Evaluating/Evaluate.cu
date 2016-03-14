#include "Base.cuh"

#include <unistd.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/MeshCrew.h"
#include "DataStructures/QualityHistogram.h"


__device__ float tetQuality(const Tet& tet);
__device__ float priQuality(const Pri& pri);
__device__ float hexQuality(const Hex& hex);


#define MIN_MAX 2147483647
#define MEAN_MAX (MIN_MAX / (blockDim.x * 3))

__global__ void evaluateCudaMeshQualityMain(int* qualMin, int* means)
{
    uint vId = threadIdx.x + blockIdx.x * blockDim.x;
    uint gid = blockIdx.x;


    if(vId < tets_length)
    {
        float q = tetQuality(tets[vId]);
        atomicMin(qualMin, int(q * MIN_MAX));
        atomicAdd(&means[gid], int(q * MEAN_MAX + 0.5));
    }

    if(vId < pris_length)
    {
        float q = priQuality(pris[vId]);
        atomicMin(qualMin, int(q * MIN_MAX));
        atomicAdd(&means[gid], int(q * MEAN_MAX + 0.5));
    }

    if(vId < hexs_length)
    {
        float q = hexQuality(hexs[vId]);
        atomicMin(qualMin, int(q * MIN_MAX));
        atomicAdd(&means[gid], int(q * MEAN_MAX + 0.5));
    }
}


// CUDA Drivers
void evaluateCudaMeshQuality(
        double meanScaleFactor,
        size_t workgroupSize,
        size_t workgroupCount,
        QualityHistogram& histogram)
{
    int* d_qualMin;
    int h_qualMin = MIN_MAX;
    cudaMalloc(&d_qualMin, sizeof(d_qualMin));
    cudaMemcpy(d_qualMin, &h_qualMin, sizeof(h_qualMin), cudaMemcpyHostToDevice);


    int* d_means = nullptr;
    int* h_means = new int[workgroupCount];
    size_t meansSize = sizeof(int) * workgroupCount;
    for(int i=0; i < workgroupCount; ++i)
        h_means[i] = 0;

    cudaMalloc(&d_means, meansSize);
    cudaMemcpy(d_means, h_means, meansSize, cudaMemcpyHostToDevice);

    cudaCheckErrors("CUDA error before evaluation");
    evaluateCudaMeshQualityMain<<<workgroupCount, workgroupSize>>>(d_qualMin, d_means);
    cudaCheckErrors("CUDA error in evaluation");

    cudaMemcpy(&h_qualMin, d_qualMin, sizeof(h_qualMin), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_means, d_means, meansSize, cudaMemcpyDeviceToHost);


    // Get minimum quality
    histogram.setMinimumQuality(h_qualMin / double(MIN_MAX));

    // Combine workgroups' mean
    double qualSum = 0.0;
    for(int i=0; i < workgroupCount; ++i)
        qualSum += h_means[i];
    histogram.setAverageQuality(
        qualSum / meanScaleFactor);
    delete[] h_means;

    // Free CUDA memory
    cudaFree(d_means);
    cudaFree(d_qualMin);
}
