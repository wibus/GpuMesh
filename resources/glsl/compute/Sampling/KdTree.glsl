struct KdNode
{
    int left;
    int right;

    uint tetBeg;
    uint tetEnd;

    vec4 separator;

    mat4 metric;
};


layout(shared, binding = KD_NODES_BUFFER_BINDING) buffer KdNodes
{
    KdNode kdNodes[];
};

subroutine mat3 metricAtSub(in vec3 position, inout uint cachedRefTet);
layout(location=METRIC_AT_SUBROUTINE_LOC)
subroutine uniform metricAtSub metricAtUni;

mat3 metricAt(in vec3 position, inout uint cachedRefTet)
{
    return metricAtUni(position, cachedRefTet);
}


layout(index=METRIC_AT_SUBROUTINE_IDX) subroutine(metricAtSub)
mat3 metricAtImpl(in vec3 position, inout uint cachedRefTet)
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
