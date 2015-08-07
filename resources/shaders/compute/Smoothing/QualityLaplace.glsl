layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform float MoveCoeff;

vec3 snapToBoundary(int boundaryID, vec3 pos);


const uint PROPOSITION_COUNT = 4;
const uint MAX_PROPOSITION_COUNT = 4;

vec3 findPatchCenter(in uint v, in Topo topo);

void testTetPropositions(
        uint vertId,
        Tet elem,
        in vec3 propositions[MAX_PROPOSITION_COUNT],
        inout float propQualities[MAX_PROPOSITION_COUNT],
        in uint propositionCount);

void testPriPropositions(
        uint vertId,
        Pri elem,
        in vec3 propositions[MAX_PROPOSITION_COUNT],
        inout float propQualities[MAX_PROPOSITION_COUNT],
        in uint propositionCount);

void testHexPropositions(
        uint vertId,
        Hex elem,
        in vec3 propositions[MAX_PROPOSITION_COUNT],
        inout float propQualities[MAX_PROPOSITION_COUNT],
        in uint propositionCount);

void main()
{
    uint uid = gl_GlobalInvocationID.x;

    if(uid >= verts.length())
        return;

    Topo topo = topos[uid];
    if(topo.type == TOPO_FIXED)
        return;

    if(topo.neigElemCount == 0)
        return;


    // Compute patch center
    vec3 pos = vec3(verts[uid].p);
    vec3 patchCenter = findPatchCenter(uid, topo);
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

    float patchQualities[PROPOSITION_COUNT] = float[](1.0, 1.0, 1.0, 1.0);


    // Compute proposition's patch quality
    uint neigElemCount = topo.neigElemCount;
    for(uint i=0, n = topo.neigElemBase; i < neigElemCount; ++i, ++n)
    {
        NeigElem neigElem = neigElems[n];
        switch(neigElem.type)
        {
        case TET_ELEMENT_TYPE:
            testTetPropositions(
                uid,
                tets[neigElem.id],
                propositions,
                patchQualities,
                PROPOSITION_COUNT);
            break;

        case PRI_ELEMENT_TYPE:
            testPriPropositions(
                uid,
                pris[neigElem.id],
                propositions,
                patchQualities,
                PROPOSITION_COUNT);
            break;

        case HEX_ELEMENT_TYPE:
            testHexPropositions(
                uid,
                hexs[neigElem.id],
                propositions,
                patchQualities,
                PROPOSITION_COUNT);
            break;
        }
    }

    // Find best proposition based on patch quality
    uint bestProposition = 0;
    float bestQualityResult = patchQualities[0];
    for(uint p=1; p < PROPOSITION_COUNT; ++p)
    {
        if(bestQualityResult < patchQualities[p])
        {
            bestProposition = p;
            bestQualityResult = patchQualities[p];
        }
    }

    // Update vertex's position
    verts[uid].p = vec4(propositions[bestProposition], 0.0);
}
