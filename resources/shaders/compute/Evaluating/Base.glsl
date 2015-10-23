///////////////////////////////
//   Function declarations   //
///////////////////////////////

// Externally defined
float tetQuality(in vec3 vp[TET_VERTEX_COUNT]);
float priQuality(in vec3 vp[PRI_VERTEX_COUNT]);
float hexQuality(in vec3 vp[HEX_VERTEX_COUNT]);

// Internally defined
float tetQuality(in Tet tet);
float priQuality(in Pri pri);
float hexQuality(in Hex hex);

float patchQuality(in uint vId);


//////////////////////////////
//   Function definitions   //
//////////////////////////////

// Element Quality
float tetQuality(in Tet tet)
{
    const vec3 vp[] = vec3[](
        verts[tet.v[0]].p,
        verts[tet.v[1]].p,
        verts[tet.v[2]].p,
        verts[tet.v[3]].p
    );

    return tetQuality(vp);
}

float priQuality(in Pri pri)
{
    const vec3 vp[] = vec3[](
        verts[pri.v[0]].p,
        verts[pri.v[1]].p,
        verts[pri.v[2]].p,
        verts[pri.v[3]].p,
        verts[pri.v[4]].p,
        verts[pri.v[5]].p
    );

    return priQuality(vp);
}

float hexQuality(in Hex hex)
{
    const vec3 vp[] = vec3[](
        verts[hex.v[0]].p,
        verts[hex.v[1]].p,
        verts[hex.v[2]].p,
        verts[hex.v[3]].p,
        verts[hex.v[4]].p,
        verts[hex.v[5]].p,
        verts[hex.v[6]].p,
        verts[hex.v[7]].p
    );

    return hexQuality(vp);
}

void accumulatePatchQuality(
        inout float patchQuality,
        inout float patchWeight,
        in float elemQuality)
{
    patchQuality = min(
        min(patchQuality, elemQuality),  // If sign(patch) != sign(elem)
        min(patchQuality * elemQuality,  // If sign(patch) & sign(elem) > 0
            patchQuality + elemQuality));// If sign(patch) & sign(elem) < 0
}

float finalizePatchQuality(in float patchQuality, in float patchWeight)
{
    return patchQuality;
}

float patchQuality(in uint vId)
{
    Topo topo = topos[vId];

    float patchWeight = 0.0;
    float patchQuality = 1.0;
    uint neigElemCount = topo.neigElemCount;
    for(uint i=0, n = topo.neigElemBase; i < neigElemCount; ++i, ++n)
    {
        NeigElem neigElem = neigElems[n];
        switch(neigElem.type)
        {
        case TET_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                tetQuality(tets[neigElem.id]));
            break;

        case PRI_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                priQuality(pris[neigElem.id]));
            break;

        case HEX_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                hexQuality(hexs[neigElem.id]));
            break;
        }
    }

    return finalizePatchQuality(patchQuality, patchWeight);
}