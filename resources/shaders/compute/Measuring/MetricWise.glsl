// Element quality interface
float tetQuality(in Tet tet);
float priQuality(in Pri pri);
float hexQuality(in Hex hex);

// Subroutine declarations
void accumulatePatchQuality(inout float patchQuality, inout float patchWeight, in float elemQuality);
float finalizePatchQuality(in float patchQuality, in float patchWeight);


// Measuring Framework
float computeLocalElementSize(in uint vId)
{
    return 1.0;
}

vec3 computeVertexEquilibrium(in uint vId)
{
    return vec3(verts[vId].p);
}

float computePatchQuality(in uint vId)
{
    return 1.0;
}
