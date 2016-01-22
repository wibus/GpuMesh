#include "../Mesh.cuh"

__device__ mat3 vertMetric(const vec3& position);

__device__ mat3 metricAt(const vec3& position)
{
    return vertMetric(position);
}
