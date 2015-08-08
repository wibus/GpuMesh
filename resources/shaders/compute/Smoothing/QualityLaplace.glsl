layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform float MoveCoeff;


// Shape measures
float tetQuality(Tet tet);
float priQuality(Pri pri);
float hexQuality(Hex hex);

// Boundaries
vec3 snapToBoundary(int boundaryID, vec3 pos);

// Optimization helper functions
vec3 computePatchCenter(in uint vId);
float computePatchQuality(in uint vId);


const uint PROPOSITION_COUNT = 4;


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

    if(topo.type > 0)
        for(uint p=1; p < PROPOSITION_COUNT; ++p)
            propositions[p] = snapToBoundary(topo.type, propositions[p]);



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
