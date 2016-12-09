const uint POSITION_THREAD_COUNT = 8;
const uint ELEMENT_THREAD_COUNT = 32;
const uint ELEMENT_SLOT_COUNT = 96;

const uint GRAD_SAMP_COUNT = 6;
const uint LINE_SAMP_COUNT = 8;

layout (local_size_x = POSITION_THREAD_COUNT, local_size_y = ELEMENT_THREAD_COUNT, local_size_z = 1) in;

struct PatchElem
{
    uint type;
    uint n;

    Tet tet;
    Pri pri;
    Hex hex;
    vec3 p[HEX_VERTEX_COUNT];
};

// Independent group range
uniform int GroupBase;
uniform int GroupSize;

uniform int SecurityCycleCount;
uniform float LocalSizeToNodeShift;

shared float nodeShift;
shared vec3 lineShift;
shared PatchElem patchElems[ELEMENT_SLOT_COUNT];
shared float elemQual[POSITION_THREAD_COUNT][ELEMENT_SLOT_COUNT];
shared float patchQual[POSITION_THREAD_COUNT];


// Smoothing Helper
float computeLocalElementSize(in uint vId);
float tetQuality(in vec3 vp[TET_VERTEX_COUNT], inout Tet tet);
float priQuality(in vec3 vp[PRI_VERTEX_COUNT], inout Pri pri);
float hexQuality(in vec3 vp[HEX_VERTEX_COUNT], inout Hex hex);
float finalizePatchQuality(
        in double patchQuality,
        in double patchWeight);
void accumulatePatchQuality(
        inout double patchQuality,
        inout double patchWeight,
        in double elemQuality);


// ENTRY POINT //
void smoothVert(uint vId)
{
    const vec3 GRAD_SAMPS[GRAD_SAMP_COUNT] = vec3[](
        vec3(-1, 0, 0), vec3( 1, 0, 0), vec3(0, -1, 0),
        vec3(0,  1, 0), vec3(0, 0, -1), vec3(0, 0,  1)
    );

    const float LINE_SAMPS[LINE_SAMP_COUNT] = float[](
        -0.25, 0.00, 0.25, 0.50,
         0.75, 1.00, 1.25, 1.50
    );

    uint pId = gl_LocalInvocationID.x;
    uint eId = gl_LocalInvocationID.y;

    Topo topo = topos[vId];
    uint neigElemCount = topo.neigElemCount;
    uint eBeg = (eId * neigElemCount) / ELEMENT_THREAD_COUNT;
    uint eEnd = ((eId+1) * neigElemCount) / ELEMENT_THREAD_COUNT;

    if(pId == 0)
    {
        for(uint e = eBeg; e < eEnd; ++e)
        {
            NeigElem elem = neigElems[topo.neigElemBase + e];
            patchElems[e].type = elem.type;
            patchElems[e].n = 0;

            switch(patchElems[e].type)
            {
            case TET_ELEMENT_TYPE :
                patchElems[e].tet = tets[elem.id];
                patchElems[e].p[0] = verts[patchElems[e].tet.v[0]].p;
                patchElems[e].p[1] = verts[patchElems[e].tet.v[1]].p;
                patchElems[e].p[2] = verts[patchElems[e].tet.v[2]].p;
                patchElems[e].p[3] = verts[patchElems[e].tet.v[3]].p;

                if(patchElems[e].tet.v[1] == vId) patchElems[e].n = 1;
                else if(patchElems[e].tet.v[2] == vId) patchElems[e].n = 2;
                else if(patchElems[e].tet.v[3] == vId) patchElems[e].n = 3;
                break;

            case PRI_ELEMENT_TYPE :
                patchElems[e].pri = pris[elem.id];
                patchElems[e].p[0] = verts[patchElems[e].pri.v[0]].p;
                patchElems[e].p[1] = verts[patchElems[e].pri.v[1]].p;
                patchElems[e].p[2] = verts[patchElems[e].pri.v[2]].p;
                patchElems[e].p[3] = verts[patchElems[e].pri.v[3]].p;
                patchElems[e].p[4] = verts[patchElems[e].pri.v[4]].p;
                patchElems[e].p[5] = verts[patchElems[e].pri.v[5]].p;

                if(patchElems[e].pri.v[1] == vId) patchElems[e].n = 1;
                else if(patchElems[e].pri.v[2] == vId) patchElems[e].n = 2;
                else if(patchElems[e].pri.v[3] == vId) patchElems[e].n = 3;
                else if(patchElems[e].pri.v[4] == vId) patchElems[e].n = 4;
                else if(patchElems[e].pri.v[5] == vId) patchElems[e].n = 5;
                break;

            case HEX_ELEMENT_TYPE :
                patchElems[e].hex = hexs[elem.id];
                patchElems[e].p[0] = verts[patchElems[e].hex.v[0]].p;
                patchElems[e].p[1] = verts[patchElems[e].hex.v[1]].p;
                patchElems[e].p[2] = verts[patchElems[e].hex.v[2]].p;
                patchElems[e].p[3] = verts[patchElems[e].hex.v[3]].p;
                patchElems[e].p[4] = verts[patchElems[e].hex.v[4]].p;
                patchElems[e].p[5] = verts[patchElems[e].hex.v[5]].p;
                patchElems[e].p[6] = verts[patchElems[e].hex.v[6]].p;
                patchElems[e].p[7] = verts[patchElems[e].hex.v[7]].p;

                if(patchElems[e].hex.v[1] == vId) patchElems[e].n = 1;
                else if(patchElems[e].hex.v[2] == vId) patchElems[e].n = 2;
                else if(patchElems[e].hex.v[3] == vId) patchElems[e].n = 3;
                else if(patchElems[e].hex.v[4] == vId) patchElems[e].n = 4;
                else if(patchElems[e].hex.v[5] == vId) patchElems[e].n = 5;
                else if(patchElems[e].hex.v[6] == vId) patchElems[e].n = 6;
                else if(patchElems[e].hex.v[7] == vId) patchElems[e].n = 7;
                break;
            }
        }
    }

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

        if(pId < GRAD_SAMP_COUNT)
        {
            for(uint e = eBeg; e < eEnd; ++e)
            {
                vec3 newPos = pos + GRAD_SAMPS[pId] * nodeShift;

                switch(patchElems[e].type)
                {
                case TET_ELEMENT_TYPE :
                    vec3 tetPos[TET_VERTEX_COUNT] = vec3[](
                        patchElems[e].p[0],
                        patchElems[e].p[1],
                        patchElems[e].p[2],
                        patchElems[e].p[3]
                    );
                    tetPos[patchElems[e].n] = newPos;
                    elemQual[pId][e] = tetQuality(tetPos, patchElems[e].tet);
                    break;
                case PRI_ELEMENT_TYPE :
                    vec3 priPos[PRI_VERTEX_COUNT] = vec3[](
                        patchElems[e].p[0],
                        patchElems[e].p[1],
                        patchElems[e].p[2],
                        patchElems[e].p[3],
                        patchElems[e].p[4],
                        patchElems[e].p[5]
                    );
                    priPos[patchElems[e].n] = newPos;
                    elemQual[pId][e] = priQuality(priPos, patchElems[e].pri);
                    break;
                case HEX_ELEMENT_TYPE :
                    vec3 hexPos[HEX_VERTEX_COUNT] = vec3[](
                        patchElems[e].p[0],
                        patchElems[e].p[1],
                        patchElems[e].p[2],
                        patchElems[e].p[3],
                        patchElems[e].p[4],
                        patchElems[e].p[5],
                        patchElems[e].p[6],
                        patchElems[e].p[7]
                    );
                    hexPos[patchElems[e].n] = newPos;
                    elemQual[pId][e] = hexQuality(hexPos, patchElems[e].hex);
                    break;
                }
            }
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

        for(uint e = eBeg; e < eEnd; ++e)
        {
            vec3 newPos = pos + lineShift * LINE_SAMPS[pId];

            switch(patchElems[e].type)
            {
            case TET_ELEMENT_TYPE :
                vec3 tetPos[TET_VERTEX_COUNT] = vec3[](
                    patchElems[e].p[0],
                    patchElems[e].p[1],
                    patchElems[e].p[2],
                    patchElems[e].p[3]
                );
                tetPos[patchElems[e].n] = newPos;
                elemQual[pId][e] = tetQuality(tetPos, patchElems[e].tet);
                break;
            case PRI_ELEMENT_TYPE :
                vec3 priPos[PRI_VERTEX_COUNT] = vec3[](
                    patchElems[e].p[0],
                    patchElems[e].p[1],
                    patchElems[e].p[2],
                    patchElems[e].p[3],
                    patchElems[e].p[4],
                    patchElems[e].p[5]
                );
                priPos[patchElems[e].n] = newPos;
                elemQual[pId][e] = priQuality(priPos, patchElems[e].pri);
                break;
            case HEX_ELEMENT_TYPE :
                vec3 hexPos[HEX_VERTEX_COUNT] = vec3[](
                    patchElems[e].p[0],
                    patchElems[e].p[1],
                    patchElems[e].p[2],
                    patchElems[e].p[3],
                    patchElems[e].p[4],
                    patchElems[e].p[5],
                    patchElems[e].p[6],
                    patchElems[e].p[7]
                );
                hexPos[patchElems[e].n] = newPos;
                elemQual[pId][e] = hexQuality(hexPos, patchElems[e].hex);
                break;
            }
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
            verts[vId].p = pos + lineShift * LINE_SAMPS[bestProposition];

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
