const uint POSITION_THREAD_COUNT = 8;
const uint ELEMENT_THREAD_COUNT = 32;
const uint ELEMENT_SLOT_COUNT = 96;

const uint GRAD_SAMP_COUNT = 6;
const uint LINE_SAMP_COUNT = 8;

const int MIN_MAX = 2147483647;

layout (local_size_x = POSITION_THREAD_COUNT, local_size_y = ELEMENT_THREAD_COUNT, local_size_z = 1) in;


// Independent group range
uniform int GroupBase;
uniform int GroupSize;

uniform int SecurityCycleCount;
uniform float LocalSizeToNodeShift;

shared float nodeShift;
shared PatchElem patchElems[ELEMENT_SLOT_COUNT];
shared int patchMin[POSITION_THREAD_COUNT];
shared float patchMean[POSITION_THREAD_COUNT];
shared float patchQual[POSITION_THREAD_COUNT];


// Smoothing Helper
float computeLocalElementSize(in uint vId);
float tetQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Tet tet);
float priQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Pri pri);
float hexQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Hex hex);
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
    uint nBeg = topo.neigElemBase + eBeg;

    if(pId == 0)
    {
        for(uint e = eBeg, ne = nBeg; e < eEnd; ++e, ++ne)
        {
            NeigElem elem = neigElems[ne];
            patchElems[e].type = elem.type;
            patchElems[e].n = elem.vId;

            switch(patchElems[e].type)
            {
            case TET_ELEMENT_TYPE :
                patchElems[e].tet = tets[elem.id];
                patchElems[e].p[0] = verts[patchElems[e].tet.v[0]].p;
                patchElems[e].p[1] = verts[patchElems[e].tet.v[1]].p;
                patchElems[e].p[2] = verts[patchElems[e].tet.v[2]].p;
                patchElems[e].p[3] = verts[patchElems[e].tet.v[3]].p;
                break;

            case PRI_ELEMENT_TYPE :
                patchElems[e].pri = pris[elem.id];
                patchElems[e].p[0] = verts[patchElems[e].pri.v[0]].p;
                patchElems[e].p[1] = verts[patchElems[e].pri.v[1]].p;
                patchElems[e].p[2] = verts[patchElems[e].pri.v[2]].p;
                patchElems[e].p[3] = verts[patchElems[e].pri.v[3]].p;
                patchElems[e].p[4] = verts[patchElems[e].pri.v[4]].p;
                patchElems[e].p[5] = verts[patchElems[e].pri.v[5]].p;
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
                break;
            }
        }
    }

    if(eId == 0)
    {
        patchMin[pId] = MIN_MAX;
        patchMean[pId] = 0.0;
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
            vec3 gradSamp = GRAD_SAMPS[pId] * nodeShift;

            for(uint e = eBeg; e < eEnd; ++e)
            {
                vec3 vertPos[HEX_VERTEX_COUNT] = vec3[](
                    patchElems[e].p[0],
                    patchElems[e].p[1],
                    patchElems[e].p[2],
                    patchElems[e].p[3],
                    patchElems[e].p[4],
                    patchElems[e].p[5],
                    patchElems[e].p[6],
                    patchElems[e].p[7]
                );

                vertPos[patchElems[e].n] = pos + gradSamp;

                float qual = 0.0;
                switch(patchElems[e].type)
                {
                case TET_ELEMENT_TYPE :
                    qual = tetQuality(vertPos, patchElems[e].tet);
                    break;
                case PRI_ELEMENT_TYPE :
                    qual = priQuality(vertPos, patchElems[e].pri);
                    break;
                case HEX_ELEMENT_TYPE :
                    qual = hexQuality(vertPos, patchElems[e].hex);
                    break;
                }

                atomicMin(patchMin[pId], int(qual * MIN_MAX));
                atomicAdd(patchMean[pId], 1.0 / qual);
            }
        }

        barrier();


        if(eId == 0)
        {
            if(patchMin[pId] <= 0.0)
                patchQual[pId] = patchMin[pId] / float(MIN_MAX);
            else
                patchQual[pId] = neigElemCount / patchMean[pId];

            patchMin[pId] = MIN_MAX;
            patchMean[pId] = 0.0;
        }

        barrier();


        vec3 gradQ = vec3(
            patchQual[1] - patchQual[0],
            patchQual[3] - patchQual[2],
            patchQual[5] - patchQual[4]);
        float gradQNorm = length(gradQ);

        vec3 lineShift;
        if(gradQNorm != 0)
            lineShift = gradQ * (nodeShift / gradQNorm);
        else
            break;


        vec3 lineSamp = lineShift * LINE_SAMPS[pId];

        for(uint e = eBeg; e < eEnd; ++e)
        {
            vec3 vertPos[HEX_VERTEX_COUNT] = vec3[](
                patchElems[e].p[0],
                patchElems[e].p[1],
                patchElems[e].p[2],
                patchElems[e].p[3],
                patchElems[e].p[4],
                patchElems[e].p[5],
                patchElems[e].p[6],
                patchElems[e].p[7]
            );

            vertPos[patchElems[e].n] = pos + lineSamp;

            float qual = 0.0;
            switch(patchElems[e].type)
            {
            case TET_ELEMENT_TYPE :
                qual = tetQuality(vertPos, patchElems[e].tet);
                break;
            case PRI_ELEMENT_TYPE :
                qual = priQuality(vertPos, patchElems[e].pri);
                break;
            case HEX_ELEMENT_TYPE :
                qual = hexQuality(vertPos, patchElems[e].hex);
                break;
            }

            atomicMin(patchMin[pId], int(qual * MIN_MAX));
            atomicAdd(patchMean[pId], 1.0 / qual);
        }

        barrier();


        if(eId == 0)
        {
            if(patchMin[pId] <= 0.0)
                patchQual[pId] = patchMin[pId] / float(MIN_MAX);
            else
                patchQual[pId] = neigElemCount / patchMean[pId];

            patchMin[pId] = MIN_MAX;
            patchMean[pId] = 0.0;
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
