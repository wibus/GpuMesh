#include "Base.cuh"

#include <DataStructures/GpuMesh.h>


struct LocalTet
{
    uint v[4];
    uint n[4];
};

__constant__ uint localTets_length;
__device__ LocalTet* localTets;



///////////////////////////////
//   Function declarations   //
///////////////////////////////
__device__ bool tetParams(const uint vi[4], const vec3& p, float coor[4]);

__device__ bool triIntersect(
        const vec3& v1,
        const vec3& v2,
        const vec3& v3,
        const vec3& orig,
        const vec3& dir);

__device__ mat3 localMetricAt(const vec3& position, uint& cachedRefTet)
{
    const LocalTet* tet = &localTets[cachedRefTet];

    float coor[4];
    bool isUnreachable = false;

    while(!isUnreachable && !tetParams(tet->v, position, coor))
    {
        vec3 vp[] = {
            refVerts[tet->v[0]].p,
            refVerts[tet->v[1]].p,
            refVerts[tet->v[2]].p,
            refVerts[tet->v[3]].p
        };

        vec3 orig = 0.25f * (vp[0] + vp[1] + vp[2] + vp[3]);
        vec3 dir = normalize(orig - position);

        int t = 4;
        int trialCount = -1;
        while(t == 4 && !isUnreachable)
        {
            // Find exit face
            for(t = 0; t < 4; ++t)
            {
                if(triIntersect(
                    vp[MeshTet_tris[t].v[0]],
                    vp[MeshTet_tris[t].v[1]],
                    vp[MeshTet_tris[t].v[2]],
                    orig, dir))
                {
                    if(tet->n[t] != -1)
                        tet = &localTets[tet->n[t]];
                    else
                        isUnreachable = true;

                    break;
                }
            }

            // If exit face not found
            if(t == 4)
            {
                // Start from an other position in the tet
                ++trialCount;

                // If there are still untried positions
                if(trialCount < 4)
                {
                    const float INV_MASS = 1.0 / 10.0;
                    const float WEIGHTS[] = {1.0, 2.0, 3.0, 4.0};

                    // Initialize ray from next position
                    orig = INV_MASS * (
                        WEIGHTS[(trialCount + 0) % 4] * vp[0] +
                        WEIGHTS[(trialCount + 1) % 4] * vp[1] +
                        WEIGHTS[(trialCount + 2) % 4] * vp[2] +
                        WEIGHTS[(trialCount + 3) % 4] * vp[3]);

                    dir = normalize(orig - position);
                }
                else
                {
                    isUnreachable = true;
                    break;
                }
            }
        }
    }


    if(isUnreachable)
    {
        // Clamp sample to current tet
        // It's seems to be the closest
        // we can get to the sampling point
        float sum = 0.0;
        if(coor[0] < 0.0) coor[0] = 0.0; else sum += coor[0];
        if(coor[1] < 0.0) coor[1] = 0.0; else sum += coor[1];
        if(coor[2] < 0.0) coor[2] = 0.0; else sum += coor[2];
        if(coor[3] < 0.0) coor[3] = 0.0; else sum += coor[3];
        coor[0] /= sum;
        coor[1] /= sum;
        coor[2] /= sum;
        coor[3] /= sum;
    }

    return coor[0] * mat3(refMetrics[tet->v[0]]) +
           coor[1] * mat3(refMetrics[tet->v[1]]) +
           coor[2] * mat3(refMetrics[tet->v[2]]) +
           coor[3] * mat3(refMetrics[tet->v[3]]);
}

__device__ metricAtFct localMetricAtPtr = localMetricAt;


// CUDA Drivers
void installCudaLocalSampler()
{
    metricAtFct d_metricAt = nullptr;
    cudaMemcpyFromSymbol(&d_metricAt, localMetricAtPtr, sizeof(metricAtFct));
    cudaMemcpyToSymbol(metricAt, &d_metricAt, sizeof(metricAtFct));


    if(verboseCuda)
        printf("I -> CUDA \tLocal Discritizer installed\n");

    cudaCheckErrors("CUDA error during Local Tets installation");
}


size_t d_localTetsLength = 0;
LocalTet* d_localTets = nullptr;
void updateCudaLocalTets(
        const std::vector<GpuLocalTet>& localTetsBuff)
{
    // Tetrahedra
    uint localTetsLength = localTetsBuff.size();
    size_t localTetsBuffSize = sizeof(decltype(localTetsBuff.front())) * localTetsLength;
    if(d_localTets == nullptr || d_localTetsLength != localTetsLength)
    {
        cudaFree(d_localTets);
        if(!localTetsLength) d_localTets = nullptr;
        else cudaMalloc(&d_localTets, localTetsBuffSize);
        cudaMemcpyToSymbol(localTets, &d_localTets, sizeof(d_localTets));

        d_localTetsLength = localTetsLength;
        cudaMemcpyToSymbol(localTets_length, &localTetsLength, sizeof(uint));
    }

    cudaMemcpy(d_localTets, localTetsBuff.data(), localTetsBuffSize, cudaMemcpyHostToDevice);

    if(verboseCuda)
        printf("I -> CUDA \tLocal Tets updated\n");

    cudaCheckErrors("CUDA error during Local Tets update");
}
