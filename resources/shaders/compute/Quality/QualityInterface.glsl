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

float tetVolume(in Tet tet);
float priVolume(in Pri pri);
float hexVolume(in Hex hex);

float tetVolume(in vec3 vp[TET_VERTEX_COUNT]);
float priVolume(in vec3 vp[PRI_VERTEX_COUNT]);
float hexVolume(in vec3 vp[HEX_VERTEX_COUNT]);


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


// Element Volume
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


float tetVolume(in vec3 vp[TET_VERTEX_COUNT])
{
    float detSum = 0.0;
    detSum += determinant(mat3(
        vp[0] - vp[3],
        vp[1] - vp[3],
        vp[2] - vp[3]));

    return detSum / 6.0;
}

float priVolume(in vec3 vp[PRI_VERTEX_COUNT])
{
    float detSum = 0.0;
    detSum += determinant(mat3(
        vp[4] - vp[2],
        vp[0] - vp[2],
        vp[1] - vp[2]));
    detSum += determinant(mat3(
        vp[5] - vp[2],
        vp[1] - vp[2],
        vp[3] - vp[2]));
    detSum += determinant(mat3(
        vp[4] - vp[2],
        vp[1] - vp[2],
        vp[5] - vp[2]));

    return detSum / 6.0;
}

float hexVolume(in vec3 vp[HEX_VERTEX_COUNT])
{
    float detSum = 0.0;
    detSum += determinant(mat3(
        vp[0] - vp[2],
        vp[1] - vp[2],
        vp[4] - vp[2]));
    detSum += determinant(mat3(
        vp[3] - vp[1],
        vp[2] - vp[1],
        vp[7] - vp[1]));
    detSum += determinant(mat3(
        vp[5] - vp[4],
        vp[1] - vp[4],
        vp[7] - vp[4]));
    detSum += determinant(mat3(
        vp[6] - vp[7],
        vp[2] - vp[7],
        vp[4] - vp[7]));
    detSum += determinant(mat3(
        vp[1] - vp[2],
        vp[7] - vp[2],
        vp[4] - vp[2]));

    return detSum / 6.0;
}
