#include "Base.cuh"

#include "DataStructures/GpuMesh.h"


#define METRIC_ERROR mat3(0.0)

struct KdNode
{
    int left;
    int right;

    uint tetBeg;
    uint tetEnd;

    vec4 separator;
};


__constant__ uint kdNodes_length;
__device__ KdNode* kdNodes;

__constant__ uint kdTets_length;
__device__ Tet* kdTets;

__constant__ uint kdMetrics_length;
__device__ mat4* kdMetrics;


__device__ bool tetParams(const Tet& tet, const vec3& p, float coor[4])
{
    dvec3 vp0 = dvec3(verts[tet.v[0]].p);
    dvec3 vp1 = dvec3(verts[tet.v[1]].p);
    dvec3 vp2 = dvec3(verts[tet.v[2]].p);
    dvec3 vp3 = dvec3(verts[tet.v[3]].p);

    dmat3 T = dmat3(vp0 - vp3, vp1 - vp3, vp2 - vp3);

    dvec3 y = inverse(T) * (dvec3(p) - vp3);
    coor[0] = float(y[0]);
    coor[1] = float(y[1]);
    coor[2] = float(y[2]);
    coor[3] = float(1.0 - (y[0] + y[1] + y[2]));

    const float EPSILON_IN = -1e-8;
    bool isIn = (coor[0] >= EPSILON_IN && coor[1] >= EPSILON_IN &&
                 coor[2] >= EPSILON_IN && coor[3] >= EPSILON_IN);
    return isIn;
}

__device__ mat3 kdTreeMetricAt(const vec3& position)
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

    float coor[4];
    uint tetEnd = node.tetEnd;
    for(uint t=node.tetBeg; t < tetEnd; ++t)
    {
        Tet tet = kdTets[t];
        if(tetParams(tet, position, coor))
        {
            return coor[0] * mat3(kdMetrics[tet.v[0]]) +
                   coor[1] * mat3(kdMetrics[tet.v[1]]) +
                   coor[2] * mat3(kdMetrics[tet.v[2]]) +
                   coor[3] * mat3(kdMetrics[tet.v[3]]);
        }
    }

    return mat3(100.0);

    /*
    // Outside of node's tets
    uint nearestVert = 0;
    double nearestDist = 1/0.0;
    for(uint t=node.tetBeg; t < tetEnd; ++t)
    {
        Tet tet = kdTets[t];
        if(distance(position, vec3(verts[tet.v[0]].p)) < nearestDist)
            nearestVert = tet.v[0];
        if(distance(position, vec3(verts[tet.v[1]].p)) < nearestDist)
            nearestVert = tet.v[1];
        if(distance(position, vec3(verts[tet.v[2]].p)) < nearestDist)
            nearestVert = tet.v[2];
        if(distance(position, vec3(verts[tet.v[3]].p)) < nearestDist)
            nearestVert = tet.v[3];
    }

    return mat3(kdMetrics[nearestVert]);
    //*/
}

__device__ metricAtFct kdTreeMetricAtPtr = kdTreeMetricAt;


// CUDA Drivers
void installCudaKdTreeDiscretizer()
{
    metricAtFct d_metricAt = nullptr;
    cudaMemcpyFromSymbol(&d_metricAt, kdTreeMetricAtPtr, sizeof(metricAtFct));
    cudaMemcpyToSymbol(metricAt, &d_metricAt, sizeof(metricAtFct));

    printf("I -> CUDA \tkD-Tree Discritizer installed\n");
}


class GpuKdNode;

size_t d_kdTetsLength = 0;
Topo* d_kdTets = nullptr;

size_t d_kdNodesLength = 0;
NeigVert* d_kdNodes = nullptr;

void updateCudaKdTreeStructure(
        const std::vector<GpuTet>& kdTetsBuff,
        const std::vector<GpuKdNode>& kdNodesBuff)
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
    printf("I -> CUDA \tkdTets updated\n");


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
    printf("I -> CUDA \tkdNodes updated\n");
}

size_t d_kdMetricsLength = 0;
GLuint* d_kdMetrics = nullptr;
void updateCudaKdTreeMetrics(
        const std::vector<glm::mat4>& kdMetricsBuff)
{
    // Group members
    uint kdMetricsLength = kdMetricsBuff.size();
    size_t kdMetricsBuffSize = sizeof(decltype(kdMetricsBuff.front())) * kdMetricsLength;
    if(d_kdMetrics == nullptr || d_kdMetricsLength != kdMetricsLength)
    {
        cudaFree(d_kdMetrics);
        if(!kdMetricsLength) d_kdMetrics = nullptr;
        else cudaMalloc(&d_kdMetrics, kdMetricsBuffSize);
        cudaMemcpyToSymbol(kdMetrics, &d_kdMetrics, sizeof(d_kdMetrics));

        d_kdMetricsLength = kdMetricsLength;
        cudaMemcpyToSymbol(kdMetrics_length, &kdMetricsLength, sizeof(uint));
    }

    cudaMemcpy(d_kdMetrics, kdMetricsBuff.data(), kdMetricsBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tkdMetrics updated\n");
}
