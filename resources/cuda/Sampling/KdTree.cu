#include "Base.cuh"

#include "DataStructures/GpuMesh.h"


struct KdNode
{
    int left;
    int right;

    uint tetBeg;
    uint tetEnd;

    vec4 separator;
};

__constant__ uint kdTets_length;
__device__ Tet* kdTets;

__constant__ uint kdNodes_length;
__device__ KdNode* kdNodes;


///////////////////////////////
//   Function declarations   //
///////////////////////////////
__device__ bool tetParams(const uint vi[4], const vec3& p, float coor[4]);


//////////////////////////////
//   Function definitions   //
//////////////////////////////
__device__ mat3 kdTreeMetricAt(const vec3& position, uint cacheId)
{
    int nodeId = 0;
    int childId = 0;
    while(childId != -1)
    {
        nodeId = childId;
        KdNode node = kdNodes[nodeId];

        float dist = node.separator.w;
        vec3 axis = vec3(node.separator);
        bool side = dot(position, axis) - dist >= 0.0;
        childId = int(mix(node.left, node.right, side));
    }


    KdNode node = kdNodes[nodeId];

    uint nodeSmallestIdx = 0;
    float nodeSmallestVal = -1/0.0;
    float nodeSmallestCoor[4];

    float coor[4];
    uint tetEnd = node.tetEnd;
    for(uint t=node.tetBeg; t < tetEnd; ++t)
    {
        Tet tet = kdTets[t];
        if(tetParams(tet.v, position, coor))
        {
            return coor[0] * mat3(refMetrics[tet.v[0]]) +
                   coor[1] * mat3(refMetrics[tet.v[1]]) +
                   coor[2] * mat3(refMetrics[tet.v[2]]) +
                   coor[3] * mat3(refMetrics[tet.v[3]]);
        }
        else
        {
            float tetSmallest = 0.0;
            if(coor[0] < tetSmallest) tetSmallest = coor[0];
            if(coor[1] < tetSmallest) tetSmallest = coor[1];
            if(coor[2] < tetSmallest) tetSmallest = coor[2];
            if(coor[3] < tetSmallest) tetSmallest = coor[3];

            if(tetSmallest > nodeSmallestVal)
            {
                nodeSmallestIdx = t;
                nodeSmallestVal = tetSmallest;
                nodeSmallestCoor[0] = coor[0];
                nodeSmallestCoor[1] = coor[1];
                nodeSmallestCoor[2] = coor[2];
                nodeSmallestCoor[3] = coor[3];
            }
        }
    }


    // Clamp coordinates for project
    float coorSum = 0.0;
    if(nodeSmallestCoor[0] < 0.0)
        nodeSmallestCoor[0] = 0.0;
    coorSum += nodeSmallestCoor[0];
    if(nodeSmallestCoor[1] < 0.0)
        nodeSmallestCoor[1] = 0.0;
    coorSum += nodeSmallestCoor[1];
    if(nodeSmallestCoor[2] < 0.0)
        nodeSmallestCoor[2] = 0.0;
    coorSum += nodeSmallestCoor[2];
    if(nodeSmallestCoor[3] < 0.0)
        nodeSmallestCoor[3] = 0.0;
    coorSum += nodeSmallestCoor[3];

    nodeSmallestCoor[0] /= coorSum;
    nodeSmallestCoor[1] /= coorSum;
    nodeSmallestCoor[2] /= coorSum;
    nodeSmallestCoor[3] /= coorSum;


    // Return projected metric
    Tet tet = kdTets[nodeSmallestIdx];
    return nodeSmallestCoor[0] * mat3(refMetrics[tet.v[0]]) +
           nodeSmallestCoor[1] * mat3(refMetrics[tet.v[1]]) +
           nodeSmallestCoor[2] * mat3(refMetrics[tet.v[2]]) +
           nodeSmallestCoor[3] * mat3(refMetrics[tet.v[3]]);
}

__device__ metricAtFct kdTreeMetricAtPtr = kdTreeMetricAt;


// CUDA Drivers
void installCudaKdTreeSampler()
{
    metricAtFct d_metricAt = nullptr;
    cudaMemcpyFromSymbol(&d_metricAt, kdTreeMetricAtPtr, sizeof(metricAtFct));
    cudaMemcpyToSymbol(metricAt, &d_metricAt, sizeof(metricAtFct));


    if(verboseCuda)
        printf("I -> CUDA \tkD-Tree Discritizer installed\n");
}


size_t d_kdTetsLength = 0;
Topo* d_kdTets = nullptr;
void updateCudaKdTets(
        const std::vector<GpuTet>& kdTetsBuff)
{
    // Tetrahedra
    uint kdTetsLength = kdTetsBuff.size();
    size_t kdTetsBuffSize = sizeof(decltype(kdTetsBuff.front())) * kdTetsLength;
    if(d_kdTets == nullptr || d_kdTetsLength != kdTetsLength)
    {
        cudaFree(d_kdTets);
        if(!kdTetsLength) d_kdTets = nullptr;
        else cudaMalloc(&d_kdTets, kdTetsBuffSize);
        cudaMemcpyToSymbol(kdTets, &d_kdTets, sizeof(d_kdTets));

        d_kdTetsLength = kdTetsLength;
        cudaMemcpyToSymbol(kdTets_length, &kdTetsLength, sizeof(uint));
    }

    cudaMemcpy(d_kdTets, kdTetsBuff.data(), kdTetsBuffSize, cudaMemcpyHostToDevice);

    if(verboseCuda)
        printf("I -> CUDA \tkdTets updated\n");
}


size_t d_kdNodesLength = 0;
KdNode* d_kdNodes = nullptr;
void updateCudaKdNodes(
        const std::vector<GpuKdNode>& kdNodesBuff)
{
    // kD-Tree Nodes
    uint kdNodesLength = kdNodesBuff.size();
    size_t kdNodesBuffSize = sizeof(decltype(kdNodesBuff.front())) * kdNodesLength;
    if(d_kdNodes == nullptr || d_kdNodesLength != kdNodesLength)
    {
        cudaFree(d_kdNodes);
        if(!kdNodesLength) d_kdNodes = nullptr;
        else cudaMalloc(&d_kdNodes, kdNodesBuffSize);
        cudaMemcpyToSymbol(kdNodes, &d_kdNodes, sizeof(d_kdNodes));

        d_kdNodesLength = kdNodesLength;
        cudaMemcpyToSymbol(kdNodes_length, &kdNodesLength, sizeof(uint));
    }

    cudaMemcpy(d_kdNodes, kdNodesBuff.data(), kdNodesBuffSize, cudaMemcpyHostToDevice);

    if(verboseCuda)
        printf("I -> CUDA \tkdNodes updated\n");
}
