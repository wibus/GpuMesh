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
    vec3 vp0 = verts[tet.v[0]].p.xyz;
    vec3 vp1 = verts[tet.v[1]].p.xyz;
    vec3 vp2 = verts[tet.v[2]].p.xyz;
    vec3 vp3 = verts[tet.v[3]].p.xyz;

    mat3 T = mat3(vp0 - vp3, vp1 - vp3, vp2 - vp3);

    vec3 y = inverse(T) * (p - vp3);
    coor[0] = y[0];
    coor[1] = y[1];
    coor[2] = y[2];
    coor[3] = 1.0 - coor[0] - coor[1] - coor[2];

    bool isIn = (coor[0] >= 0.0 && coor[1] >= 0.0 &&
                 coor[2] >= 0.0 && coor[3] >= 0.0);
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

    float coor[4];
    uint tetEnd = node.tetEnd;
    for(uint t=node.tetBeg; t < tetEnd; ++t)
    {
        Tet tet = kdTets[t];
        if(tetParams(tet, position, coor))
        {
            mat3 m = coor[0] * mat3(kdMetrics[tet.v[0]]) +
                     coor[1] * mat3(kdMetrics[tet.v[1]]) +
                     coor[2] * mat3(kdMetrics[tet.v[2]]) +
                     coor[3] * mat3(kdMetrics[tet.v[3]]);
            return m;
        }
    }

    // Outside of node's tets
    return METRIC_ERROR;
}
