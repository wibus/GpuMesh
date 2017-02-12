#include "Base.cuh"

#include "DataStructures/GpuMesh.h"


struct KdNode
{
    int left;
    int right;

    uint tetBeg;
    uint tetEnd;

    vec4 separator;

    GpuMetric metric;
};

__constant__ uint kdNodes_length;
__device__ KdNode* kdNodes;


//////////////////////////////
//   Function definitions   //
//////////////////////////////
__device__ mat3 kdTreeMetricAt(const vec3& position, uint& cachedRefTet)
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
        childId = side ? node.right : node.left;
    }


    KdNode node = kdNodes[nodeId];
    return mat3(vec3(node.metric[0]),
                vec3(node.metric[1]),
                vec3(node.metric[2]));
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
