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

layout(std140, binding = KD_METRICS_BUFFER_BINDING) buffer KdMetrics
{
    mat4 kdMetrics[];
};


subroutine mat3 metricAtSub(in vec3 position);
layout(location=0) subroutine uniform metricAtSub metricAtUni;

mat3 metricAt(in vec3 position)
{
    return metricAtUni(position);
}


bool tetParams(in Tet tet, in vec3 p, out float coor[4])
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
    coor[3] = float(1.0LF - (y[0] + y[1] + y[2]));

    const float EPSILON_IN = -1e-8;
    bool isIn = (coor[0] >= EPSILON_IN && coor[1] >= EPSILON_IN &&
                 coor[2] >= EPSILON_IN && coor[3] >= EPSILON_IN);
    return isIn;
}

layout(index=0) subroutine(metricAtSub)
mat3 metricAtImpl(in vec3 position)
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
        if(tetParams(tet, position, coor))
        {
            return coor[0] * mat3(kdMetrics[tet.v[0]]) +
                   coor[1] * mat3(kdMetrics[tet.v[1]]) +
                   coor[2] * mat3(kdMetrics[tet.v[2]]) +
                   coor[3] * mat3(kdMetrics[tet.v[3]]);
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
    return nodeSmallestCoor[0] * mat3(kdMetrics[tet.v[0]]) +
           nodeSmallestCoor[1] * mat3(kdMetrics[tet.v[1]]) +
           nodeSmallestCoor[2] * mat3(kdMetrics[tet.v[2]]) +
           nodeSmallestCoor[3] * mat3(kdMetrics[tet.v[3]]);
}
