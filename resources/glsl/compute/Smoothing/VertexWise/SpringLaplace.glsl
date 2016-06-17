uniform float MoveCoeff;


// Smoothing Helper
vec3 computeVertexEquilibrium(in uint vId);


// ENTRY POINT //
void smoothVert(uint vId)
{
    vec3 patchCenter = computeVertexEquilibrium(vId);
    verts[vId].p = mix(verts[vId].p, patchCenter, MoveCoeff);
}
