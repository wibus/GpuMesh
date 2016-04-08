#include "Base.cuh"


__device__ mat3 vertMetric(const vec3& position);

__device__ mat3 analyticMetricAt(const vec3& position, uint cacheId)
{
    return vertMetric(position);
}

__device__ metricAtFct analyticMetricAtPtr = analyticMetricAt;


// CUDA Drivers
void installCudaAnalyticSampler()
{
    metricAtFct d_metricAt = nullptr;
    cudaMemcpyFromSymbol(&d_metricAt, analyticMetricAtPtr, sizeof(metricAtFct));
    cudaMemcpyToSymbol(metricAt, &d_metricAt, sizeof(metricAtFct));


    if(verboseCuda)
        printf("I -> CUDA \tAnalytic Discritizer installed\n");
}
