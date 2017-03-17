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
    int visitedTet = 0;
    bool outOfTet = false;
    bool outOfBounds = false;

    while(!outOfBounds && !tetParams(tet->v, position, coor))
    {
        vec3 orig, dir;

        if(visitedTet == 0)
        {
            orig = 0.25f * (
                refVerts[tet->v[0]].p +
                refVerts[tet->v[1]].p +
                refVerts[tet->v[2]].p +
                refVerts[tet->v[3]].p);

            dir = normalize(orig - position);
        }


        int t=0;
        for(;t < 4; ++t)
        {
            if(triIntersect(
                refVerts[tet->v[MeshTet_tris[t].v[0]]].p,
                refVerts[tet->v[MeshTet_tris[t].v[1]]].p,
                refVerts[tet->v[MeshTet_tris[t].v[2]]].p,
                orig, dir))
            {
                if(tet->n[t] != -1)
                {
                    ++visitedTet;
                    tet = &localTets[tet->n[t]];
                }
                else
                {
                    outOfBounds = true;
                    outOfTet = true;
                }

                break;
            }
        }

        if(t == 4)
        {
            if(visitedTet == 0)
            {
                outOfTet = true;
                break;
            }
            else
            {
                visitedTet = 0;

                if(coor[0] < 0.0 && tet->n[0] != -1)
                    tet = &localTets[tet->n[0]];
                else if(coor[1] < 0.0 && tet->n[1] != -1)
                    tet = &localTets[tet->n[1]];
                else if(coor[2] < 0.0 && tet->n[2] != -1)
                    tet = &localTets[tet->n[2]];
                else if(tet->n[3] != -1)
                    tet = &localTets[tet->n[3]];
                else
                {
                    // Boundary reached
                    outOfTet = true;
                    break;
                }
            }
        }
    }

    if(outOfTet)
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
