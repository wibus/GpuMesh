///////////////////////////////
//   Function declarations   //
///////////////////////////////

// Externally defined
float tetQuality(in vec3 vp[TET_VERTEX_COUNT], in Tet tet);
float priQuality(in vec3 vp[PRI_VERTEX_COUNT], in Pri pri);
float hexQuality(in vec3 vp[HEX_VERTEX_COUNT], in Hex hex);

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

    return tetQuality(vp, tet);
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

    return priQuality(vp, pri);
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

    return hexQuality(vp, hex);
}

void accumulatePatchQuality(
        inout double patchQuality,
        inout double patchWeight,
        in double elemQuality)
{
    patchQuality = min(patchQuality, elemQuality);
    /*
    patchQuality = min(
        min(patchQuality, elemQuality),  // If sign(patch) != sign(elem)
        min(patchQuality * elemQuality,  // If sign(patch) & sign(elem) > 0
            patchQuality + elemQuality));// If sign(patch) & sign(elem) < 0
            */
}

float finalizePatchQuality(in double patchQuality, in double patchWeight)
{
    return float(patchQuality);
    /*
    double s = sign(patchQuality);
    return float(s * sqrt(s*patchQuality));
    */
}

float patchQuality(in uint vId)
{
    Topo topo = topos[vId];

    double patchWeight = 0.0;
    double patchQuality = 1.0;
    uint neigElemCount = topo.neigElemCount;
    for(uint i=0, n = topo.neigElemBase; i < neigElemCount; ++i, ++n)
    {
        NeigElem neigElem = neigElems[n];

        switch(neigElem.type)
        {
        case TET_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                double(tetQuality(tets[neigElem.id])));
            break;

        case PRI_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                double(priQuality(pris[neigElem.id])));
            break;

        case HEX_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                double(hexQuality(hexs[neigElem.id])));
            break;
        }
    }

    return finalizePatchQuality(patchQuality, patchWeight);
}
