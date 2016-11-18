#include "Base.cuh"


__device__ mat3 uniformMetricAt(const vec3& position, uint& cachedRefTet)
{
    return mat3(MetricScaling * MetricScaling);
}

__device__ metricAtFct uniformMetricAtPtr = uniformMetricAt;


// CUDA Drivers
void installCudaUniformSampler()
{
    metricAtFct d_metricAt = nullptr;
    cudaMemcpyFromSymbol(&d_metricAt, uniformMetricAtPtr, sizeof(metricAtFct));
    cudaMemcpyToSymbol(metricAt, &d_metricAt, sizeof(metricAtFct));


    if(verboseCuda)
        printf("I -> CUDA \tUniform Discritizer installed\n");
}
