uniform float MoveCoeff;


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
    const uint PROPOSITION_COUNT = 8;
    const float OFFSETS[PROPOSITION_COUNT] = float[](
        -0.25, 0.00, 0.10, 0.20,
         0.40, 0.80, 1.20, 1.60
    );

    vec3 shift = centerDist * MoveCoeff;
    vec3 propositions[PROPOSITION_COUNT] = vec3[](
        pos + shift * OFFSETS[0],
        pos + shift * OFFSETS[1],
        pos + shift * OFFSETS[2],
        pos + shift * OFFSETS[3],
        pos + shift * OFFSETS[4],
        pos + shift * OFFSETS[5],
        pos + shift * OFFSETS[6],
        pos + shift * OFFSETS[7]
    );


    // Choose best position based on quality geometric mean
    uint bestProposition = 0;
    float bestQualityMean = -1.0/0.0; // -Inf
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
