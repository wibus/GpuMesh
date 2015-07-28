float tetQuality(in vec3 vp[4]);
float priQuality(in vec3 vp[6]);
float hexQuality(in vec3 vp[8]);


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
