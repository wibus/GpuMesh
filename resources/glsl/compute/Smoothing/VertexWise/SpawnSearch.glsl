uniform float MoveCoeff;

const uint PROPOSITION_COUNT = 4;


// Smoothing helper
vec3 computeVertexEquilibrium(in uint vId);
float patchQuality(in uint vId);


// ENTRY POINT //
void smoothVert(uint vId)
{
    // Compute patch center
    vec3 pos = verts[vId].p;

    // Update vertex's position
    verts[vId].p = pos;
}
