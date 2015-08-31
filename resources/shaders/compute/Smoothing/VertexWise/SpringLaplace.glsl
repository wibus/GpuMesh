uniform float MoveCoeff;


// Boundaries
vec3 snapToBoundary(int boundaryID, vec3 pos);

// Smoothing Helper
vec3 computePatchCenter(in uint vId);


// ENTRY POINT //
void smoothVertex(uint vId)
{
    vec3 patchCenter = computePatchCenter(vId);

    vec3 pos = vec3(verts[vId].p);
    pos = mix(pos, patchCenter, MoveCoeff);


    Topo topo = topos[vId];
    if(topo.type > 0)
    {
        pos = snapToBoundary(topo.type, pos);
    }


    // Write
    verts[vId].p = vec4(pos, 0.0);
}
