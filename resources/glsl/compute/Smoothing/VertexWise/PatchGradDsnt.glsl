const uint POSITION_THREAD_COUNT = 8;
const uint ELEMENT_THREAD_COUNT = 32;
const uint ELEMENT_SLOT_COUNT = 96;

layout (local_size_x = POSITION_THREAD_COUNT, local_size_y = ELEMENT_THREAD_COUNT, local_size_z = 1) in;

uniform float MoveCoeff;
uniform int SecurityCycleCount;
uniform float LocalSizeToNodeShift;

// Independent group range
uniform int GroupBase;
uniform int GroupSize;

shared float nodeShift;
shared vec3 lineShift;
shared float elemQual[POSITION_THREAD_COUNT][ELEMENT_SLOT_COUNT];
shared float patchQual[POSITION_THREAD_COUNT];
shared vec3 propositions[POSITION_THREAD_COUNT];

const uint GRAD_SAMP_COUNT = 6;
const vec3 GRAD_SAMPS[GRAD_SAMP_COUNT] = vec3[](
    vec3(-1, 0, 0),
    vec3( 1, 0, 0),
    vec3(0, -1, 0),
    vec3(0,  1, 0),
    vec3(0, 0, -1),
    vec3(0, 0,  1)
);

const uint LINE_SAMP_COUNT = 8;
const float LINE_SAMPS[LINE_SAMP_COUNT] = float[](
    -0.25,
     0.00,
     0.25,
     0.50,
     0.75,
     1.00,
     1.25,
     1.50
);

// Smoothing Helper
float computeLocalElementSize(in uint vId);
float patchQuality(in uint vId);


// Externally defined
float tetQuality(in vec3 vp[TET_VERTEX_COUNT], inout Tet tet);
float finalizePatchQuality(in double patchQuality, in double patchWeight);
void accumulatePatchQuality(
        inout double patchQuality,
        inout double patchWeight,
        in double elemQuality);


// ENTRY POINT //
void smoothVert(uint vId)
{
    uint pId = gl_LocalInvocationID.x;
    uint eId = gl_LocalInvocationID.y;

    Topo topo = topos[vId];
    uint neigElemCount = topo.neigElemCount;
    NeigElem elem = neigElems[topo.neigElemBase + eId];

    Tet tetElem = tets[elem.id];
    vec3[TET_VERTEX_COUNT] tetVerts = vec3[](
        verts[tetElem.v[0]].p,
        verts[tetElem.v[1]].p,
        verts[tetElem.v[2]].p,
        verts[tetElem.v[3]].p
    );

    uint nId = 3;
    if(tetElem.v[0] == vId) nId = 0;
    else if(tetElem.v[1] == vId) nId = 1;
    else if(tetElem.v[2] == vId) nId = 2;

    if(pId == 0 && eId == 0)
    {
        // Compute local element size
        float localSize = computeLocalElementSize(vId);

        // Initialize node shift distance
        nodeShift = localSize * LocalSizeToNodeShift;
    }

    barrier();

    float originalNodeShift = nodeShift;
    for(int c=0; c < SecurityCycleCount; ++c)
    {
        vec3 pos = verts[vId].p;

        if(eId < neigElemCount && pId < GRAD_SAMP_COUNT)
        {
            // Define patch quality gradient samples
            tetVerts[nId] = pos + GRAD_SAMPS[pId] * nodeShift;
            elemQual[pId][eId] = tetQuality(tetVerts, tetElem);
        }

        barrier();

        if(eId == 0 && pId < GRAD_SAMP_COUNT)
        {
            double patchWeight = 0.0;
            double patchQuality = 0.0;
            for(uint e = 0; e < neigElemCount; ++e)
                accumulatePatchQuality(
                    patchQuality, patchWeight,
                    double(elemQual[pId][e]));

            patchQual[pId] = finalizePatchQuality(patchQuality, patchWeight);
        }

        barrier();

        if(eId == 0 && pId == 0)
        {
            vec3 gradQ = vec3(
                patchQual[1] - patchQual[0],
                patchQual[3] - patchQual[2],
                patchQual[5] - patchQual[4]);
            float gradQNorm = length(gradQ);

            if(gradQNorm != 0)
                lineShift = gradQ * (nodeShift / gradQNorm);
            else
                lineShift = vec3(0.0);
        }

        barrier();

        if(lineShift == vec3(0.0))
            break;

        if(eId < neigElemCount)
        {
            tetVerts[nId] = pos + lineShift * LINE_SAMPS[pId];
            elemQual[pId][eId] = tetQuality(tetVerts, tetElem);
        }

        barrier();

        if(eId == 0)
        {
            double patchWeight = 0.0;
            double patchQuality = 0.0;
            for(uint e = 0; e < neigElemCount; ++e)
                accumulatePatchQuality(
                    patchQuality, patchWeight,
                    double(elemQual[pId][e]));

            patchQual[pId] = finalizePatchQuality(patchQuality, patchWeight);
            propositions[pId] = tetVerts[nId];
        }

        barrier();

        if(eId == 0 && pId == 0)
        {
            uint bestProposition = 0;
            float bestQualityMean = patchQual[0];
            for(uint p=1; p < LINE_SAMP_COUNT; ++p)
            {
                if(patchQual[p] > bestQualityMean)
                {
                    bestQualityMean = patchQual[p];
                    bestProposition = p;
                }
            }

            // Update vertex's position
            verts[vId].p = propositions[bestProposition];

            // Scale node shift and stop if it is too small
            nodeShift *= abs(LINE_SAMPS[bestProposition]);
        }

        barrier();

        if(nodeShift < originalNodeShift / 10.0)
            break;
    }
}


void main()
{
    if(gl_WorkGroupID.x < GroupSize)
    {
        uint idx = GroupBase + gl_WorkGroupID.x;
        uint vId = groupMembers[idx];
        smoothVert(vId);
    }
}
