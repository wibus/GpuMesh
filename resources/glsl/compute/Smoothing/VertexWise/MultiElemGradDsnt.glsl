const uint NODE_THREAD_COUNT = 32;
const uint ELEMENT_THREAD_COUNT = 8;
const uint POSITION_SLOT_COUNT = 8;

const uint GRAD_SAMP_COUNT = 6;
const uint LINE_SAMP_COUNT = 8;

const int MIN_MAX = 2147483647;


// Independent group range
uniform int GroupBase;
uniform int GroupSize;

uniform int SecurityCycleCount;
uniform float LocalSizeToNodeShift;

shared float nodeShift[NODE_THREAD_COUNT];
shared int patchMin[NODE_THREAD_COUNT][POSITION_SLOT_COUNT];
shared float patchMean[NODE_THREAD_COUNT][POSITION_SLOT_COUNT];
shared float patchQual[NODE_THREAD_COUNT][POSITION_SLOT_COUNT];


// Smoothing Helper
uint getInvocationVertexId();
bool isSmoothableVertex(uint vId);
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
        -0.25, 0.00, 0.10, 0.20,
         0.40, 0.80, 1.20, 1.60
    );

    uint eId = gl_LocalInvocationID.x;
    uint nId = gl_LocalInvocationID.y;

    Topo topo = topos[vId];
    uint neigBase = topo.neigElemBase;
    uint neigElemCount = topo.neigElemCount;
    uint nBeg = neigBase + (eId * neigElemCount) / ELEMENT_THREAD_COUNT;
    uint nEnd = neigBase + ((eId+1) * neigElemCount) / ELEMENT_THREAD_COUNT;


    if(eId < POSITION_SLOT_COUNT)
    {
        patchMin[nId][eId] = MIN_MAX;
        patchMean[nId][eId] = 0.0;
    }

    if(eId == 0)
    {
        // Compute local element size
        float localSize = computeLocalElementSize(vId);

        // Initialize node shift distance
        nodeShift[nId] = localSize * LocalSizeToNodeShift;
    }

    barrier();


    float originalNodeShift = nodeShift[nId];
    for(int c=0; c < SecurityCycleCount; ++c)
    {
        vec3 pos = verts[vId].p;

        for(uint e = nBeg; e < nEnd; ++e)
        {
            NeigElem elem = neigElems[e];
            vec3 vertPos[HEX_VERTEX_COUNT];

            float qual = 0.0;
            switch(elem.type)
            {
            case TET_ELEMENT_TYPE :
                vertPos[0] = verts[tets[elem.id].v[0]].p;
                vertPos[1] = verts[tets[elem.id].v[1]].p;
                vertPos[2] = verts[tets[elem.id].v[2]].p;
                vertPos[3] = verts[tets[elem.id].v[3]].p;
                break;

            case PRI_ELEMENT_TYPE :
                vertPos[0] = verts[pris[elem.id].v[0]].p;
                vertPos[1] = verts[pris[elem.id].v[1]].p;
                vertPos[2] = verts[pris[elem.id].v[2]].p;
                vertPos[3] = verts[pris[elem.id].v[3]].p;
                vertPos[4] = verts[pris[elem.id].v[4]].p;
                vertPos[5] = verts[pris[elem.id].v[5]].p;
                break;

            case HEX_ELEMENT_TYPE :
                vertPos[0] = verts[hexs[elem.id].v[0]].p;
                vertPos[1] = verts[hexs[elem.id].v[1]].p;
                vertPos[2] = verts[hexs[elem.id].v[2]].p;
                vertPos[3] = verts[hexs[elem.id].v[3]].p;
                vertPos[4] = verts[hexs[elem.id].v[4]].p;
                vertPos[5] = verts[hexs[elem.id].v[5]].p;
                vertPos[6] = verts[hexs[elem.id].v[6]].p;
                vertPos[7] = verts[hexs[elem.id].v[7]].p;
                break;
            }

            for(uint p=0; p < GRAD_SAMP_COUNT; ++p)
            {
                vec3 gradSamp = pos + GRAD_SAMPS[p] * nodeShift[nId];
                vertPos[elem.vId] = gradSamp;

                float qual = 0.0;
                switch(elem.type)
                {
                case TET_ELEMENT_TYPE :
                    qual = tetQuality(vertPos, tets[elem.id]);
                    break;

                case PRI_ELEMENT_TYPE :
                    qual = priQuality(vertPos, pris[elem.id]);
                    break;

                case HEX_ELEMENT_TYPE :
                    qual = hexQuality(vertPos, hexs[elem.id]);
                    break;
                }

                atomicMin(patchMin[nId][p], int(qual * MIN_MAX));
                atomicAdd(patchMean[nId][p], 1.0 / qual);
            }
        }

        barrier();


        if(eId < GRAD_SAMP_COUNT)
        {
            if(patchMin[nId][eId] <= 0.0)
                patchQual[nId][eId] = patchMin[nId][eId] / float(MIN_MAX);
            else
                patchQual[nId][eId] = neigElemCount / patchMean[nId][eId];

            patchMin[nId][eId] = MIN_MAX;
            patchMean[nId][eId] = 0.0;
        }

        barrier();


        vec3 gradQ = vec3(
            patchQual[nId][1] - patchQual[nId][0],
            patchQual[nId][3] - patchQual[nId][2],
            patchQual[nId][5] - patchQual[nId][4]);
        float gradQNorm = length(gradQ);

        vec3 lineShift;
        if(gradQNorm != 0)
            lineShift = gradQ * (nodeShift[nId] / gradQNorm);
        else
            break;


        for(uint e = nBeg; e < nEnd; ++e)
        {
            NeigElem elem = neigElems[e];
            vec3 vertPos[HEX_VERTEX_COUNT];

            switch(elem.type)
            {
            case TET_ELEMENT_TYPE :
                vertPos[0] = verts[tets[elem.id].v[0]].p;
                vertPos[1] = verts[tets[elem.id].v[1]].p;
                vertPos[2] = verts[tets[elem.id].v[2]].p;
                vertPos[3] = verts[tets[elem.id].v[3]].p;
                break;

            case PRI_ELEMENT_TYPE :
                vertPos[0] = verts[pris[elem.id].v[0]].p;
                vertPos[1] = verts[pris[elem.id].v[1]].p;
                vertPos[2] = verts[pris[elem.id].v[2]].p;
                vertPos[3] = verts[pris[elem.id].v[3]].p;
                vertPos[4] = verts[pris[elem.id].v[4]].p;
                vertPos[5] = verts[pris[elem.id].v[5]].p;
                break;

            case HEX_ELEMENT_TYPE :
                vertPos[0] = verts[hexs[elem.id].v[0]].p;
                vertPos[1] = verts[hexs[elem.id].v[1]].p;
                vertPos[2] = verts[hexs[elem.id].v[2]].p;
                vertPos[3] = verts[hexs[elem.id].v[3]].p;
                vertPos[4] = verts[hexs[elem.id].v[4]].p;
                vertPos[5] = verts[hexs[elem.id].v[5]].p;
                vertPos[6] = verts[hexs[elem.id].v[6]].p;
                vertPos[7] = verts[hexs[elem.id].v[7]].p;
                break;
            }

            for(uint p=0; p < LINE_SAMP_COUNT; ++p)
            {
                vec3 lineSamp = pos + lineShift * LINE_SAMPS[p];
                vertPos[elem.vId] = lineSamp;

                float qual = 0.0;
                switch(elem.type)
                {
                case TET_ELEMENT_TYPE :
                    qual = tetQuality(vertPos, tets[elem.id]);
                    break;

                case PRI_ELEMENT_TYPE :
                    qual = priQuality(vertPos, pris[elem.id]);
                    break;

                case HEX_ELEMENT_TYPE :
                    qual = hexQuality(vertPos, hexs[elem.id]);
                    break;
                }

                atomicMin(patchMin[nId][p], int(qual * MIN_MAX));
                atomicAdd(patchMean[nId][p], 1.0 / qual);
            }
        }

        barrier();


        if(eId < LINE_SAMP_COUNT)
        {
            if(patchMin[nId][eId] <= 0.0)
                patchQual[nId][eId] = patchMin[nId][eId] / float(MIN_MAX);
            else
                patchQual[nId][eId] = neigElemCount / patchMean[nId][eId];

            patchMin[nId][eId] = MIN_MAX;
            patchMean[nId][eId] = 0.0;
        }

        barrier();


        if(eId == 0)
        {
            uint bestProposition = 0;
            float bestQualityMean = patchQual[nId][0];
            for(uint p=1; p < LINE_SAMP_COUNT; ++p)
            {
                if(patchQual[nId][p] > bestQualityMean)
                {
                    bestQualityMean = patchQual[nId][p];
                    bestProposition = p;
                }
            }

            // Update vertex's position
            verts[vId].p = pos + lineShift * LINE_SAMPS[bestProposition];

            // Scale node shift and stop if it is too small
            nodeShift[nId] *= abs(LINE_SAMPS[bestProposition]);
        }

        barrier();


        if(nodeShift[nId] < originalNodeShift / 10.0)
            break;
    }
}


void main()
{
    uint localId = gl_WorkGroupID.x * gl_LocalGroupSizeARB.y + gl_LocalInvocationID.y;

    if(localId < GroupSize)
    {
        uint idx = GroupBase + localId;
        uint vId = groupMembers[idx];
        smoothVert(vId);
    }
}
