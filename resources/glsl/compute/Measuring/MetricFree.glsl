// Element Volume
float riemannianDistance(in vec3 a, in vec3 b)
{
    return distance(a, b);
}

vec3 riemannianSegment(in vec3 a, in vec3 b)
{
    return b - a;
}

float tetVolume(in vec3 vp[TET_VERTEX_COUNT])
{
    float detSum = 0.0;
    detSum += determinant(mat3(
        vp[3] - vp[0],
        vp[3] - vp[1],
        vp[3] - vp[2]));

    return detSum / 6.0;
}

float priVolume(in vec3 vp[PRI_VERTEX_COUNT])
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


    return detSum / 6.0;
}

float hexVolume(in vec3 vp[HEX_VERTEX_COUNT])
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

    return detSum / 6.0;
}


// High level measurement
vec3 computeVertexEquilibrium(in uint vId)
{
    Topo topo = topos[vId];

    uint totalVertCount = 0;
    vec3 patchCenter = vec3(0.0);
    uint neigElemCount = topo.neigElemCount;
    for(uint i=0, n = topo.neigElemBase; i<neigElemCount; ++i, ++n)
    {
        NeigElem neigElem = neigElems[n];

        switch(neigElem.type)
        {
        case TET_ELEMENT_TYPE:
            totalVertCount += TET_VERTEX_COUNT - 1;
            for(uint i=0; i < TET_VERTEX_COUNT; ++i)
                patchCenter += vec3(verts[tets[neigElem.id].v[i]].p);
            break;

        case PRI_ELEMENT_TYPE:
            totalVertCount += PRI_VERTEX_COUNT - 1;
            for(uint i=0; i < PRI_VERTEX_COUNT; ++i)
                patchCenter += vec3(verts[pris[neigElem.id].v[i]].p);
            break;

        case HEX_ELEMENT_TYPE:
            totalVertCount += HEX_VERTEX_COUNT - 1;
            for(uint i=0; i < HEX_VERTEX_COUNT; ++i)
                patchCenter += vec3(verts[hexs[neigElem.id].v[i]].p);
            break;
        }
    }

    vec3 pos = vec3(verts[vId].p);
    patchCenter = (patchCenter - pos * float(neigElemCount))
                    / float(totalVertCount);
    return patchCenter;
}
