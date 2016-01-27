#include "../Mesh.cuh"


// Metric sampling function
typedef mat3 (*metricAtFct)(const vec3&);
extern __device__ metricAtFct metricAt;
