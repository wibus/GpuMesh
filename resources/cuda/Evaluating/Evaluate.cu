#include "Base.cuh"

#include <unistd.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/MeshCrew.h"
#include "DataStructures/QualityHistogram.h"


__device__ float tetQuality(const Tet& tet);
__device__ float priQuality(const Pri& pri);
__device__ float hexQuality(const Hex& hex);


__device__ int qualMin;
__device__ int* means;

__constant__ uint hists_length;
__device__ int* hists;


#define MIN_MAX 2147483647
#define MEAN_MAX (MIN_MAX / (blockDim.x * 3))

__device__ void commit(uint gid, float q)
{
    atomicMin(&qualMin, int(q * MIN_MAX));
    atomicAdd(&means[gid], int(q * MEAN_MAX + 0.5));

    int bucket = int(max(q * hists_length, 0.0));
    atomicAdd(&hists[bucket], 1);
}

__global__ void evaluateCudaMeshQualityMain()
{
    uint vId = threadIdx.x + blockIdx.x * blockDim.x;
    uint gid = blockIdx.x;


    if(vId < tets_length)
    {
        commit( gid, tetQuality(tets[vId]) );
    }

    if(vId < pris_length)
    {
        commit( gid, priQuality(pris[vId]) );
    }

    if(vId < hexs_length)
    {
        commit( gid, hexQuality(hexs[vId]) );
    }
}


// CUDA Drivers
void evaluateCudaMeshQuality(
        double meanScaleFactor,
        size_t workgroupSize,
        size_t workgroupCount,
        QualityHistogram& histogram)
{
    int h_qualMin = MIN_MAX;
    cudaMemcpyToSymbol(qualMin, &h_qualMin, sizeof(qualMin));


    int* d_means = nullptr;
    int* h_means = new int[workgroupCount];
    size_t meansSize = sizeof(int) * workgroupCount;
    for(int i=0; i < workgroupCount; ++i)
        h_means[i] = 0;

    cudaMalloc(&d_means, meansSize);
    cudaMemcpy(d_means, h_means, meansSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(means, &d_means, sizeof(d_means));


    int* d_hists = nullptr;
    int* h_hists = const_cast<int*>(histogram.buckets().data());
    size_t histsSize = sizeof(int) * histogram.bucketCount();
    uint h_hists_length = histogram.bucketCount();

    cudaMalloc(&d_hists, histsSize);
    cudaMemcpy(d_hists, h_hists, histsSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(hists, &d_hists, sizeof(d_hists));
    cudaMemcpyToSymbol(hists_length, &h_hists_length, sizeof(hists_length));


    cudaCheckErrors("CUDA error before evaluation");
    evaluateCudaMeshQualityMain<<<workgroupCount, workgroupSize>>>();
    cudaCheckErrors("CUDA error in evaluation");

    cudaMemcpyFromSymbol(&h_qualMin, qualMin, sizeof(h_qualMin));
    cudaMemcpy(h_means, d_means, meansSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hists, d_hists, histsSize, cudaMemcpyDeviceToHost);

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
    cudaFree(d_hists);
}
