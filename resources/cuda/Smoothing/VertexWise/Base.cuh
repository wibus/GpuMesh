#include "../../Evaluating/Base.cuh"


// Tetrahedron smoothing function
typedef void (*smoothVertFct)(uint vId);
extern __device__ smoothVertFct smoothVert;



__device__ inline float3 toFloat3(const vec3& v)
{
    return make_float3(v.x, v.y, v.z);
}

__device__ inline vec3 toVec3(const float3& v)
{
    return vec3(v.x, v.y, v.z);
}
