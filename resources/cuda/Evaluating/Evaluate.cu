#include "Base.cuh"

#include <unistd.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/MeshCrew.h"
#include "DataStructures/QualityHistogram.h"


__device__ float tetQuality(const Tet& tet);
__device__ float priQuality(const Pri& pri);
__device__ float hexQuality(const Hex& hex);


__device__ int qualMin;
__device__ float invLogSum;

__constant__ uint hists_length;
__device__ int* hists;


#define MIN_MAX 2147483647

__device__ void commit(uint gid, float q)
{
    atomicMin(&qualMin, int(q * MIN_MAX));
    atomicAdd(&invLogSum, log(1.0 / q));

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
        size_t workgroupSize,
        size_t workgroupCount,
        QualityHistogram& histogram)
{
    int h_qualMin = MIN_MAX;
    cudaMemcpyToSymbol(qualMin, &h_qualMin, sizeof(qualMin));

    float h_invLogSum = 0.0f;
    cudaMemcpyToSymbol(invLogSum, &h_invLogSum, sizeof(invLogSum));


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
    cudaMemcpyFromSymbol(&h_invLogSum, invLogSum, sizeof(h_invLogSum));
    cudaMemcpy(h_hists, d_hists, histsSize, cudaMemcpyDeviceToHost);

    // Get minimum quality
    histogram.setMinimumQuality(h_qualMin / double(MIN_MAX));

    // Get inverse log quality sum
    histogram.setInvQualityLogSum(h_invLogSum);

    // Free CUDA memory
    cudaFree(d_hists);
}
