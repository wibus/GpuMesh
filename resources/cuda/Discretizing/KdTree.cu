#include "Base.cuh"


#define METRIC_ERROR mat3(0.0)

struct KdNode
{
    int left;
    int right;

    uint tetBeg;
    uint tetEnd;

    vec4 separator;
};


__device__ uint kdNodes_length;
__device__ KdNode* kdNodes;

__device__ uint kdTets_length;
__device__ Tet* kdTets;

__device__ uint kdMetrics_length;
__device__ mat4* kdMetrics;


__device__ bool tetParams(const Tet& tet, const vec3& p, float coor[4])
{
    dvec3 vp0 = dvec3(verts[tet.v[0]].p);
    dvec3 vp1 = dvec3(verts[tet.v[1]].p);
    dvec3 vp2 = dvec3(verts[tet.v[2]].p);
    dvec3 vp3 = dvec3(verts[tet.v[3]].p);

    dmat3 T = dmat3(vp0 - vp3, vp1 - vp3, vp2 - vp3);

    dvec3 y = inverse(T) * (glm::dvec3(p) - vp3);
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

    printf("kDTree()\n");
}
