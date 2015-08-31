uniform float MoveCoeff;

const uint PROPOSITION_COUNT = 4;


// Boundaries
vec3 snapToBoundary(int boundaryID, vec3 pos);

// Smoothing helper
vec3 computePatchCenter(in uint vId);
float computePatchQuality(in uint vId);


// ENTRY POINT //
void smoothVertex(uint vId)
{
    // Compute patch center
    vec3 patchCenter = computePatchCenter(vId);
    vec3 pos = vec3(verts[vId].p);
    vec3 centerDist = patchCenter - pos;


    // Define propositions for new vertex's position
    vec3 propositions[PROPOSITION_COUNT] = vec3[](
        pos,
        patchCenter - centerDist * MoveCoeff,
        patchCenter,
        patchCenter + centerDist * MoveCoeff
    );

    Topo topo = topos[vId];
    if(topo.type > 0)
    {
        for(uint p=1; p < PROPOSITION_COUNT; ++p)
            propositions[p] = snapToBoundary(
                topo.type, propositions[p]);
    }



    // Choose best position based on quality geometric mean
    uint bestProposition = 0;
    float bestQualityMean = 0.0;
    for(uint p=0; p < PROPOSITION_COUNT; ++p)
    {
        // Quality evaluation functions will use this updated position
        // to compute element shape measures.
        verts[vId].p = vec4(propositions[p], 0.0);

        float patchQuality = computePatchQuality(vId);

        if(patchQuality > bestQualityMean)
        {
            bestQualityMean = patchQuality;
            bestProposition = p;
        }
    }


    // Update vertex's position
    verts[vId].p = vec4(propositions[bestProposition], 0.0);
}
