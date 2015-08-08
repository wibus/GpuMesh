layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform float MoveCoeff;


// Shape measures
float tetQuality(Tet tet);
float priQuality(Pri pri);
float hexQuality(Hex hex);

// Boundaries
vec3 snapToBoundary(int boundaryID, vec3 pos);

// Optimization helper functions
vec3 findPatchCenter(in uint v, in Topo topo);
void accumulatePatchQuality(in float elemQ, inout float patchQ);
void finalizePatchQuality(inout float patchQ);


const uint PROPOSITION_COUNT = 4;


void main()
{
    uint uid = gl_GlobalInvocationID.x;

    if(uid >= verts.length())
        return;

    Topo topo = topos[uid];
    if(topo.type == TOPO_FIXED)
        return;

    uint neigElemCount = topo.neigElemCount;
    if(neigElemCount == 0)
        return;

    // Compute patch center
    vec3 patchCenter = findPatchCenter(uid, topo);
    vec3 pos = vec3(verts[uid].p);
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
        verts[uid].p = vec4(propositions[p], 0.0);


        float patchQuality = 1.0;
        for(uint i=0, n = topo.neigElemBase; i < neigElemCount; ++i, ++n)
        {
            NeigElem neigElem = neigElems[n];
            switch(neigElem.type)
            {
            case TET_ELEMENT_TYPE:
                accumulatePatchQuality(
                    tetQuality(tets[neigElem.id]),
                    patchQuality);
                break;

            case PRI_ELEMENT_TYPE:
                accumulatePatchQuality(
                    priQuality(pris[neigElem.id]),
                    patchQuality);
                break;

            case HEX_ELEMENT_TYPE:
                accumulatePatchQuality(
                    hexQuality(hexs[neigElem.id]),
                    patchQuality);
                break;
            }

            if(patchQuality <= 0.0)
            {
                break;
            }
        }

        finalizePatchQuality(patchQuality);

        if(patchQuality > bestQualityMean)
        {
            bestQualityMean = patchQuality;
            bestProposition = p;
        }
    }


    // Update vertex's position
    verts[uid].p = vec4(propositions[bestProposition], 0.0);
}
