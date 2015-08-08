layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform float MoveCoeff;


// Boundaries
vec3 snapToBoundary(int boundaryID, vec3 pos);

// Optimization helper functions
vec3 computePatchCenter(in uint vId);


void main()
{
    uint vId = gl_GlobalInvocationID.x;

    if(vId >= verts.length())
        return;

    Topo topo = topos[vId];
    if(topo.type == TOPO_FIXED)
        return;

    uint neigElemCount = topo.neigElemCount;
    if(neigElemCount == 0)
        return;


    vec3 patchCenter = computePatchCenter(vId);

    vec3 pos = vec3(verts[vId].p);
    pos = mix(pos, patchCenter, MoveCoeff);

    if(topo.type > 0)
    {
        pos = snapToBoundary(topo.type, pos);
    }


    // Write
    verts[vId].p = vec4(pos, 0.0);
}
