#include "Base.cuh"


///////////////////////////////
//   Function declarations   //
///////////////////////////////
__device__ metricAtFct metricAt = nullptr;



//////////////////////////////
//   Function definitions   //
//////////////////////////////
__device__ mat3 interpolateMetrics(const mat3& m1, const mat3& m2, float a)
{
    return mat3(
        mix(m1[0], m2[0], a),
        mix(m1[1], m2[1], a),
        mix(m1[2], m2[2], a));
}

__device__ mat3 vertMetric(const vec3& position)
{
    vec3 vp = position * vec3(7);

    float localElemSize = 0.0;
    localElemSize = 1.0 / pow(1000, 1.0/3.0);

    float elemSize = localElemSize;
    float elemSizeInv2 = 1.0 / (elemSize * elemSize);

    float scale = pow(3, sin(vp.x));
    float targetElemSizeX = elemSize * scale;
    float targetElemSizeXInv2 = 1.0 / (targetElemSizeX * targetElemSizeX);
    float targetElemSizeZ = elemSize / scale;
    float targetElemSizeZInv2 = 1.0 / (targetElemSizeZ * targetElemSizeZ);

    float rx = targetElemSizeXInv2;
    float ry = elemSizeInv2;
    float rz = elemSizeInv2;

    return mat3(
        vec3(rx, 0,  0),
        vec3(0,  ry, 0),
        vec3(0,  0,  rz));
}

__device__ mat3 vertMetric(uint vId)
{
    return vertMetric(vec3(verts[vId].p));
}

__device__ void boundingBox(vec3& minBounds, vec3& maxBounds)
{
    minBounds = vec3(1.0/0.0);
    maxBounds = vec3(-1.0/0.0);
    uint vertCount = verts_length;
    for(uint v=0; v < vertCount; ++v)
    {
        vec3 vertPos = vec3(verts[v].p);
        minBounds = min(minBounds, vertPos);
        maxBounds = max(maxBounds, vertPos);
    }
}
