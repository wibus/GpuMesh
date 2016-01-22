#include "../Mesh.cuh"

#include "DataStructures/Mesh.h"
#include "DataStructures/MeshCrew.h"

__device__ int qualMin;
__device__ int* means;

__device__ float tetQuality(const Tet &tet);
__device__ float priQuality(const Pri &pri);
__device__ float hexQuality(const Hex &hex);


#define MIN_MAX 2147483647
#define MEAN_MAX (MIN_MAX / (blockDim.x * 3))

__device__ void evaluateCudaQualityMain()
{
    uint vId = threadIdx.x + blockIdx.x * blockDim.x;
    uint gid = blockIdx.x;


    if(vId < tets_length)
    {
        float q = tetQuality(tets[vId]);
        atomicMin(&qualMin, int(q * MIN_MAX));
        atomicAdd(&means[gid], int(q * MEAN_MAX + 0.5));
    }

    if(vId < pris_length)
    {
        float q = priQuality(pris[vId]);
        atomicMin(&qualMin, int(q * MIN_MAX));
        atomicAdd(&means[gid], int(q * MEAN_MAX + 0.5));
    }

    if(vId < hexs_length)
    {
        float q = hexQuality(hexs[vId]);
        atomicMin(&qualMin, int(q * MIN_MAX));
        atomicAdd(&means[gid], int(q * MEAN_MAX + 0.5));
    }
}


// CUDA Drivers
void evaluateCudaMetricConformityQuality(
        const Mesh& mesh,
        const MeshCrew& crew)
{
}
