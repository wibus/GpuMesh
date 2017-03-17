#include "Base.cuh"


///////////////////////////////
//   Function declarations   //
///////////////////////////////
__device__ metricAtFct metricAt = nullptr;

__constant__ float MetricScaling = 1.0;
__constant__ float MetricScalingSqr = 1.0;
__constant__ float MetricScalingCube = 1.0;
__constant__ float MetricAspectRatio = 1.0;



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
    float x = position.x * (2.5 * M_PI);

    float sizeX = MetricScaling * pow(MetricAspectRatio, (1.0 - cos(x)) / 2.0);

    float Mx = sizeX * sizeX;
    float My = MetricScalingSqr;
    float Mz = My;

    return mat3(
            vec3(Mx, 0,  0),
            vec3(0,  My, 0),
            vec3(0,  0,  Mz));
}

__device__ bool tetParams(const uint vi[4], const vec3& p, float coor[4])
{
    dvec3 vp0 = dvec3(refVerts[vi[0]].p);
    dvec3 vp1 = dvec3(refVerts[vi[1]].p);
    dvec3 vp2 = dvec3(refVerts[vi[2]].p);
    dvec3 vp3 = dvec3(refVerts[vi[3]].p);

    dmat3 T = dmat3(vp0 - vp3, vp1 - vp3, vp2 - vp3);

    dvec3 y = inverse(T) * (dvec3(p) - vp3);
    coor[0] = float(y[0]);
    coor[1] = float(y[1]);
    coor[2] = float(y[2]);
    coor[3] = float(1.0 - (y[0] + y[1] + y[2]));

    const float EPSILON_IN = -1e-4;
    bool isIn = (coor[0] >= EPSILON_IN && coor[1] >= EPSILON_IN &&
                 coor[2] >= EPSILON_IN && coor[3] >= EPSILON_IN);
    return isIn;
}

__device__ bool triIntersect(
        const vec3& v1,
        const vec3& v2,
        const vec3& v3,
        const vec3& orig,
        const vec3& dir)
{
    const float EPSILON = 1e-12;

    vec3 e1 = v2 - v1;
    vec3 e2 = v3 - v1;
    vec3 pvec = cross(dir, e2);

    float det = dot(pvec, e1);
    if (det < EPSILON)
    {
        return false;
    }

    vec3 tvec = orig - v1;
    float u = dot(tvec, pvec);
    if (u < 0.0 || u > det)
    {
        return false;
    }

    vec3 qvec = cross(tvec,e1);
    float v = dot(dir, qvec);
    if (v < 0.0 || v + u > det)
    {
        return false;
    }

    return true;
}


// CUDA Drivers
void setCudaMetricScaling(double scaling)
{
    cudaCheckErrors("CUDA error before sampler scaling");

    float h_scaling = scaling;
    cudaMemcpyToSymbol(MetricScaling, &h_scaling, sizeof(h_scaling));
}

void setCudaMetricScalingSqr(double scalingSqr)
{
    float h_scalingSqr = scalingSqr;
    cudaMemcpyToSymbol(MetricScalingSqr, &h_scalingSqr, sizeof(h_scalingSqr));
}

void setCudaMetricScalingCube(double scalingCube)
{
    float h_scalingCube = scalingCube;
    cudaMemcpyToSymbol(MetricScalingCube, &h_scalingCube, sizeof(h_scalingCube));

    cudaCheckErrors("CUDA error after sampler scaling");
}

void setCudaMetricAspectRatio(double aspectRatio)
{
    float h_aspectRatio = aspectRatio;
    cudaMemcpyToSymbol(MetricAspectRatio, &h_aspectRatio, sizeof(h_aspectRatio));
}
