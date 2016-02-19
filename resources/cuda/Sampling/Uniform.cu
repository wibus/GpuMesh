#include "Base.cuh"


__device__ mat3 uniformMetricAt(const vec3& position)
{
    return mat3(0.0);
}

__device__ metricAtFct uniformMetricAtPtr = uniformMetricAt;


// CUDA Drivers
void installCudaUniformSampler()
{
    metricAtFct d_metricAt = nullptr;
    cudaMemcpyFromSymbol(&d_metricAt, uniformMetricAtPtr, sizeof(metricAtFct));
    cudaMemcpyToSymbol(metricAt, &d_metricAt, sizeof(metricAtFct));

    printf("I -> CUDA \tUniform Discritizer installed\n");
}
