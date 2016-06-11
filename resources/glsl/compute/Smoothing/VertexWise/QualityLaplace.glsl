uniform float MoveCoeff;

const uint PROPOSITION_COUNT = 4;


// Smoothing helper
vec3 computeVertexEquilibrium(in uint vId);
float patchQuality(in uint vId);


// ENTRY POINT //
void smoothVert(uint vId)
{
    // Compute patch center
    vec3 patchCenter = computeVertexEquilibrium(vId);
    vec3 pos = verts[vId].p;
    vec3 centerDist = patchCenter - pos;


    // Define propositions for new vertex's position
    vec3 propositions[PROPOSITION_COUNT] = vec3[](
        pos,
        patchCenter - centerDist * MoveCoeff,
        patchCenter,
        patchCenter + centerDist * MoveCoeff
    );


    // Choose best position based on quality geometric mean
    uint bestProposition = 0;
    float bestQualityMean = 0.0;
    for(uint p=0; p < PROPOSITION_COUNT; ++p)
    {
        // Quality evaluation functions will use this updated position
        // to compute element shape measures.
        verts[vId].p = propositions[p];

        float patchQuality = patchQuality(vId);

        if(patchQuality > bestQualityMean)
        {
            bestQualityMean = patchQuality;
            bestProposition = p;
        }
    }


    // Update vertex's position
    verts[vId].p = propositions[bestProposition];
}
