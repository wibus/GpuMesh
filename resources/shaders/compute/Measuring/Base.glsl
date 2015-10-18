///////////////////////////////
//   Function declarations   //
///////////////////////////////

// Externally defined
float tetVolume(in vec3 vp[TET_VERTEX_COUNT]);
float priVolume(in vec3 vp[PRI_VERTEX_COUNT]);
float hexVolume(in vec3 vp[HEX_VERTEX_COUNT]);

// Internally defined
float tetVolume(in Tet tet);
float priVolume(in Pri pri);
float hexVolume(in Hex hex);


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

    return tetVolume(vp);
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

    return priVolume(vp);
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

    return hexVolume(vp);
}
