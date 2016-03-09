#include "Base.cuh"

#include <DataStructures/GpuMesh.h>


struct LocalTet
{
    uint v[4];
    uint n[4];
};

__constant__ uint localTets_length;
__device__ LocalTet* localTets;

__constant__ uint localCache_length;
__device__ uint* localCache;



///////////////////////////////
//   Function declarations   //
///////////////////////////////
__device__ bool tetParams(const uint vi[4], const vec3& p, float coor[4]);


#define MAX_TABOO 32
__device__ bool isTaboo(uint tId, uint taboo[MAX_TABOO], uint count)
{
    if(tId != -1)
    {
        for(uint i=0; i < count; ++i)
            if(tId == taboo[i])
                return true;
    }

    return false;
}

__device__ mat3 localMetricAt(const vec3& position, uint cacheId)
{
    // Taboo search structures
    uint tabooCount = 0;
    uint taboo[MAX_TABOO];

    uint tetId = localCache[cacheId];
    LocalTet tet = localTets[tetId];

    float coor[4];
    while(!tetParams(tet.v, position, coor))
    {
        uint n = -1;
        float minCoor = 1/0.0;

        if(coor[0] < minCoor && !isTaboo(tet.n[0], taboo, tabooCount))
        {
            n = 0;
            minCoor = coor[0];
        }
        if(coor[1] < minCoor && !isTaboo(tet.n[1], taboo, tabooCount))
        {
            n = 1;
            minCoor = coor[1];
        }
        if(coor[2] < minCoor && !isTaboo(tet.n[2], taboo, tabooCount))
        {
            n = 2;
            minCoor = coor[2];
        }
        if(coor[3] < minCoor && !isTaboo(tet.n[3], taboo, tabooCount))
        {
            n = 3;
            minCoor = coor[3];
        }

        bool clipCurrentTet = false;
        if(n != -1)
        {
            uint nextTet = tet.n[n];

            if((nextTet != -1))
            {
                if(tabooCount < MAX_TABOO)
                {
                    // Add last tet to taboo list
                    taboo[tabooCount] = tetId;
                    ++tabooCount;

                    // Fetch the next local tet
                    tet = localTets[nextTet];
                    tetId = nextTet;
                }
                else
                {
                    // We went too far,
                    // stay where we are
                    clipCurrentTet = true;
                }
            }
            else
            {
                // The sampling point is on
                // the other side of the boundary
                clipCurrentTet = true;
                // This may not be an issue
            }
        }
        else
        {
            // Every surrounding tet
            // were already visited
            clipCurrentTet = true;
        }


        if(clipCurrentTet)
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
            break;
        }
    }

    // TODO wbussiere 2016-03-07 :
    //  Verify potential race conditions issues
    localCache[cacheId] = tetId;

    return coor[0] * mat3(refMetrics[tet.v[0]]) +
           coor[1] * mat3(refMetrics[tet.v[1]]) +
           coor[2] * mat3(refMetrics[tet.v[2]]) +
           coor[3] * mat3(refMetrics[tet.v[3]]);
}

__device__ metricAtFct localMetricAtPtr = localMetricAt;


// CUDA Drivers
void installCudaLocalSampler()
{
    metricAtFct d_metricAt = nullptr;
    cudaMemcpyFromSymbol(&d_metricAt, localMetricAtPtr, sizeof(metricAtFct));
    cudaMemcpyToSymbol(metricAt, &d_metricAt, sizeof(metricAtFct));

    printf("I -> CUDA \tLocal Discritizer installed\n");
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
    printf("I -> CUDA \tLocal Tets updated\n");
}

size_t d_localCacheLength = 0;
uint* d_localCache = nullptr;
void updateCudaLocalCache(
        const std::vector<GLuint>& localCacheBuff)
{
    // kD-Tree Nodes
    uint localCacheLength = localCacheBuff.size();
    size_t localCacheBuffSize = sizeof(decltype(localCacheBuff.front())) * localCacheLength;
    if(d_localCache == nullptr || d_localCacheLength != localCacheLength)
    {
        cudaFree(d_localCache);
        if(!localCacheLength) d_localCache = nullptr;
        else cudaMalloc(&d_localCache, localCacheBuffSize);
        cudaMemcpyToSymbol(localCache, &d_localCache, sizeof(d_localCache));

        d_localCacheLength = localCacheLength;
        cudaMemcpyToSymbol(localCache_length, &localCacheLength, sizeof(uint));
    }

    cudaMemcpy(d_localCache, localCacheBuff.data(), localCacheBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tlLocal Cache updated\n");
}
