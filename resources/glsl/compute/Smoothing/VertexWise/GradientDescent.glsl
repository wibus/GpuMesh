uniform float MoveCoeff;
uniform int SecurityCycleCount;
uniform float LocalSizeToNodeShift;

// Smoothing Helper
float computeLocalElementSize(in uint vId);
float patchQuality(in uint vId);


// ENTRY POINT //
void smoothVert(uint vId)
{
    // Compute local element size
    float localSize = computeLocalElementSize(vId);

    // Initialize node shift distance
    float nodeShift = localSize * LocalSizeToNodeShift;
    float originalNodeShift = nodeShift;

    for(int c=0; c < SecurityCycleCount; ++c)
    {
        // Define patch quality gradient samples
        vec3 pos = verts[vId].p;
        const uint GRADIENT_SAMPLE_COUNT = 6;
        float sampleQualities[GRADIENT_SAMPLE_COUNT] = float[]
                (1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        vec3 gradSamples[GRADIENT_SAMPLE_COUNT] = vec3[](
            pos + vec3(-nodeShift, 0.0,   0.0),
            pos + vec3( nodeShift, 0.0,   0.0),
            pos + vec3( 0.0,  -nodeShift, 0.0),
            pos + vec3( 0.0,   nodeShift, 0.0),
            pos + vec3( 0.0,   0.0,  -nodeShift),
            pos + vec3( 0.0,   0.0,   nodeShift)
        );

        for(uint p=0; p < GRADIENT_SAMPLE_COUNT; ++p)
        {
            // Quality evaluation functions will use this updated position
            // to compute element shape measures.
            verts[vId].p = gradSamples[p];

            // Compute patch quality
            sampleQualities[p] = patchQuality(vId);
        }
        verts[vId].p = pos;

        vec3 gradQ = vec3(
            sampleQualities[1] - sampleQualities[0],
            sampleQualities[3] - sampleQualities[2],
            sampleQualities[5] - sampleQualities[4]);
        float gradQNorm = length(gradQ);

        if(gradQNorm == 0)
            break;


        const uint PROPOSITION_COUNT = 8;
        const float OFFSETS[PROPOSITION_COUNT] = float[](
            -0.25, 0.00, 0.10, 0.20,
             0.40, 0.80, 1.20, 1.60
        );

        vec3 shift = gradQ * (nodeShift / gradQNorm);
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

        uint bestProposition = 0;
        float bestQualityMean = -1.0/0.0; // -Inf
        for(uint p=0; p < PROPOSITION_COUNT; ++p)
        {
            // Quality evaluation functions will use this updated position
            // to compute element shape measures.
            verts[vId].p = propositions[p];

            // Compute patch quality
            float patchQuality = patchQuality(vId);

            if(patchQuality > bestQualityMean)
            {
                bestQualityMean = patchQuality;
                bestProposition = p;
            }
        }


        // Update vertex's position
        verts[vId].p = propositions[bestProposition];

        // Scale node shift and stop if it is too small
        nodeShift *= abs(OFFSETS[bestProposition]);
        if(nodeShift < originalNodeShift / 10.0)
            break;
    }
}
