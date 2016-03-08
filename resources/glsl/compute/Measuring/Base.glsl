///////////////////////////////
//   Function declarations   //
///////////////////////////////

// Externally defined
float tetVolume(in vec3 vp[TET_VERTEX_COUNT], in Tet tet);
float priVolume(in vec3 vp[PRI_VERTEX_COUNT], in Pri pri);
float hexVolume(in vec3 vp[HEX_VERTEX_COUNT], in Hex hex);

// Internally defined
float tetVolume(in Tet tet);
float priVolume(in Pri pri);
float hexVolume(in Hex hex);

float computeLocalElementSize(in uint vId);


//////////////////////////////
//   Function definitions   //
//////////////////////////////
float tetVolume(in Tet tet)
{
    const vec3 vp[] = vec3[](
        verts[tet.v[0]].p,
        verts[tet.v[1]].p,
        verts[tet.v[2]].p,
        verts[tet.v[3]].p
    );

    return tetVolume(vp, tet);
}

float priVolume(in Pri pri)
{
    const vec3 vp[] = vec3[](
        verts[pri.v[0]].p,
        verts[pri.v[1]].p,
        verts[pri.v[2]].p,
        verts[pri.v[3]].p,
        verts[pri.v[4]].p,
        verts[pri.v[5]].p
    );

    return priVolume(vp, pri);
}

float hexVolume(in Hex hex)
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

    return hexVolume(vp, hex);
}

float computeLocalElementSize(in uint vId)
{
    vec3 pos = vec3(verts[vId].p);
    Topo topo = topos[vId];

    float totalSize = 0.0;
    uint neigVertCount = topo.neigVertCount;
    for(uint i=0, n = topo.neigVertBase; i < neigVertCount; ++i, ++n)
    {
        totalSize += length(pos - vec3(verts[neigVerts[n].v].p));
    }

    return totalSize / neigVertCount;
}
