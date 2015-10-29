// Element Volume
float tetVolume(in vec3 vp[TET_VERTEX_COUNT])
{
    return 1.0;
}

float priVolume(in vec3 vp[PRI_VERTEX_COUNT])
{
    return 1.0;
}

float hexVolume(in vec3 vp[HEX_VERTEX_COUNT])
{
    return 1.0;
}


// High level measurement
vec3 computeVertexEquilibrium(in uint vId)
{
    return vec3(verts[vId].p);
}
