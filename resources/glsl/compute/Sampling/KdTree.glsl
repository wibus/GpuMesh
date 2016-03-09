struct KdNode
{
    int left;
    int right;

    uint tetBeg;
    uint tetEnd;

    vec4 separator;
};


layout(shared, binding = KD_NODES_BUFFER_BINDING) buffer KdNodes
{
    KdNode kdNodes[];
};

layout(shared, binding = KD_TETS_BUFFER_BINDING) buffer KdTets
{
    Tet kdTets[];
};

bool tetParams(in uint vi[4], in vec3 p, out float coor[4]);


subroutine mat3 metricAtSub(in vec3 position, in uint cacheId);
layout(location=0) subroutine uniform metricAtSub metricAtUni;

mat3 metricAt(in vec3 position, in uint cacheId)
{
    return metricAtUni(position, cacheId);
}


layout(index=0) subroutine(metricAtSub)
mat3 metricAtImpl(in vec3 position, in uint cacheId)
{
    const mat3 METRIC_ERROR = mat3(0.0);

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
