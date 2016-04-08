#include "Base.cuh"


__device__ mat3 dummyMetricAt(const vec3& position, uint cacheId)
{
    return mat3(1.0);
}

__device__ metricAtFct dummyMetricAtPtr = dummyMetricAt;


// CUDA Drivers
void installCudaDummySampler()
{
    metricAtFct d_metricAt = nullptr;
    cudaMemcpyFromSymbol(&d_metricAt, dummyMetricAtPtr, sizeof(metricAtFct));
    cudaMemcpyToSymbol(metricAt, &d_metricAt, sizeof(metricAtFct));


    if(verboseCuda)
        printf("I -> CUDA \tDummy Discritizer installed\n");
}
