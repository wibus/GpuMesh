struct KdNode
{
    uint left;
    uint right;

    uint tetBeg;
    uint tetEnd;

    vec4 separator;
    vec4 minBox;
    vec4 maxBox;
};


layout(shared, binding = KD_NODES_BUFFER_BINDING) buffer KdNodes
{
    KdNode kdNodes[];
};

layout(shared, binding = KD_TETS_BUFFER_BINDING) buffer KdTets
{
    Tet kdTets[];
};

layout(shared, binding = KD_METRICS_BUFFER_BINDING) buffer KdMetrics
{
    mat4 kdMetrics[];
};

const mat3 METRIC_ERROR = mat3(0.0);


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
    if(kdNodes.length() == 0)
        return METRIC_ERROR;

    KdNode node = kdNodes[0];
    while(node.left  != 0 &&
          node.right != 0)
    {
        float dist = node.separator.w;
        vec3 axis = vec3(node.separator);
        if(dot(position, axis) - dist < 0.0)
            node = kdNodes[node.left];
        else
            node = kdNodes[node.right];
    }

    float coor[4];
    uint tetEnd = node.tetEnd;
    for(uint t=node.tetBeg; t < tetEnd; ++t)
    {
        Tet tet = kdTets[t];
        if(tetParams(tet, position, coor))
        {
            return METRIC_ERROR;
            mat3 m = coor[0] * mat3(kdMetrics[tet.v[0]]) +
                     coor[1] * mat3(kdMetrics[tet.v[1]]) +
                     coor[2] * mat3(kdMetrics[tet.v[2]]) +
                     coor[3] * mat3(kdMetrics[tet.v[3]]);
            return m;
        }
    }

    return mat3(1.0);
}
