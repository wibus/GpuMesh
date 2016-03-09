#include "Base.cuh"

#include <DataStructures/GpuMesh.h>


__device__ mat3 localMetricAt(const vec3& position)
{
    return mat3(0.0);
}

__device__ metricAtFct localMetricAtPtr = localMetricAt;


// CUDA Drivers
void installCudaLocalSampler()
{
    metricAtFct d_metricAt = nullptr;
    cudaMemcpyFromSymbol(&d_metricAt, localMetricAtPtr, sizeof(metricAtFct));
    cudaMemcpyToSymbol(metricAt, &d_metricAt, sizeof(metricAtFct));

    printf("I -> CUDA \tLocal Discritizer installed\n");
}

void updateCudaLocalTets(
        const std::vector<GpuLocalTet>& localTetsBuff)
{

}

void updateCudaLocalCache(
        const std::vector<GLuint>& localCacheBuff)
{

}
