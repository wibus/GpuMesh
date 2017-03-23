uniform float MetricScaling;
uniform float MetricScalingSqr;
uniform float MetricScalingCube;

// Element Volume
float riemannianDistance(in vec3 a, in vec3 b, inout uint cachedRefTet)
{
    return distance(a, b) * MetricScaling;
}

vec3 riemannianSegment(in vec3 a, in vec3 b, inout uint cachedRefTet)
{
    return (b - a) * MetricScaling;
}

float tetVolume(in vec3 vp[PARAM_VERTEX_COUNT], inout Tet tet)
{
    float detSum = 0.0;
    detSum += determinant(mat3(
        vp[3] - vp[0],
        vp[3] - vp[1],
        vp[3] - vp[2]));

    return MetricScalingCube * detSum / 6.0;
}

float priVolume(in vec3 vp[PARAM_VERTEX_COUNT], inout Pri pri)
{
    vec3 e02 = vp[2] - vp[0];
    vec3 e12 = vp[2] - vp[1];
    vec3 e32 = vp[2] - vp[3];
    vec3 e42 = vp[2] - vp[4];
    vec3 e52 = vp[2] - vp[5];

    float detSum = 0.0;
    detSum += determinant(mat3(e32, e52, e42));
    detSum += determinant(mat3(e02, e32, e42));
    detSum += determinant(mat3(e12, e02, e42));


    return MetricScalingCube * detSum / 6.0;
}

float hexVolume(in vec3 vp[PARAM_VERTEX_COUNT], inout Hex hex)
{
    float detSum = 0.0;
    detSum += determinant(mat3(
        vp[0] - vp[1],
        vp[0] - vp[4],
        vp[0] - vp[3]));
    detSum += determinant(mat3(
        vp[2] - vp[1],
        vp[2] - vp[3],
        vp[2] - vp[6]));
    detSum += determinant(mat3(
        vp[5] - vp[4],
        vp[5] - vp[1],
        vp[5] - vp[6]));
    detSum += determinant(mat3(
        vp[7] - vp[4],
        vp[7] - vp[6],
        vp[7] - vp[3]));
    detSum += determinant(mat3(
        vp[4] - vp[1],
        vp[4] - vp[6],
        vp[4] - vp[3]));

    return MetricScalingCube * detSum / 6.0;
}


// High level measurement
vec3 computeVertexEquilibrium(in uint vId)
{
    Topo topo = topos[vId];

    float totalWeight = 0.0f;
    vec3 displacement = vec3(0.0);
    vec3 pos = verts[vId].p;

    uint neigElemCount = topo.neigElemCount;
    for(uint i=0, n = topo.neigElemBase; i<neigElemCount; ++i, ++n)
    {
        NeigElem neigElem = neigElems[n];

        switch(neigElem.type)
        {
        case TET_ELEMENT_TYPE:
            totalWeight += TET_VERTEX_COUNT - 1;
            for(uint i=0; i < TET_VERTEX_COUNT; ++i)
                displacement += verts[tets[neigElem.id].v[i]].p - pos;
            break;

        case PRI_ELEMENT_TYPE:
            totalWeight += PRI_VERTEX_COUNT - 1;
            for(uint i=0; i < PRI_VERTEX_COUNT; ++i)
                displacement += verts[pris[neigElem.id].v[i]].p - pos;
            break;

        case HEX_ELEMENT_TYPE:
            totalWeight += HEX_VERTEX_COUNT - 1;
            for(uint i=0; i < HEX_VERTEX_COUNT; ++i)
                displacement += verts[hexs[neigElem.id].v[i]].p - pos;
            break;
        }
    }

    return pos + displacement / totalWeight;
}
