const uint NODE_THREAD_COUNT = 8;
const uint ELEMENT_THREAD_COUNT = 32;
const uint ELEMENT_PER_THREAD_COUNT = 3;
const uint POSITION_SLOT_COUNT = 8;

const uint GRAD_SAMP_COUNT = 6;
const uint LINE_SAMP_COUNT = 8;

const int MIN_MAX = 2147483647;

layout (local_size_x = NODE_THREAD_COUNT, local_size_y = ELEMENT_THREAD_COUNT, local_size_z = 1) in;


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
        -0.25, 0.00, 0.25, 0.50,
         0.75, 1.00, 1.25, 1.50
    );

    uint nId = gl_LocalInvocationID.x;
    uint eId = gl_LocalInvocationID.y;

    Topo topo = topos[vId];
    uint neigElemCount = topo.neigElemCount;
    uint eBeg = (eId * neigElemCount) / ELEMENT_THREAD_COUNT;
    uint eEnd = ((eId+1) * neigElemCount) / ELEMENT_THREAD_COUNT;

    PatchElem elems[ELEMENT_PER_THREAD_COUNT];
    for(uint e=0, id = eBeg; id < eEnd; ++e, ++id)
    {
        NeigElem elem = neigElems[topo.neigElemBase + id];
        elems[e].type = elem.type;
        elems[e].n = 0;

        switch(elems[e].type)
        {
        case TET_ELEMENT_TYPE :
            elems[e].tet = tets[elem.id];
            elems[e].p[0] = verts[elems[e].tet.v[0]].p;
            elems[e].p[1] = verts[elems[e].tet.v[1]].p;
            elems[e].p[2] = verts[elems[e].tet.v[2]].p;
            elems[e].p[3] = verts[elems[e].tet.v[3]].p;

            if(elems[e].tet.v[1] == vId) elems[e].n = 1;
            else if(elems[e].tet.v[2] == vId) elems[e].n = 2;
            else if(elems[e].tet.v[3] == vId) elems[e].n = 3;
            break;

        case PRI_ELEMENT_TYPE :
            elems[e].pri = pris[elem.id];
            elems[e].p[0] = verts[elems[e].pri.v[0]].p;
            elems[e].p[1] = verts[elems[e].pri.v[1]].p;
            elems[e].p[2] = verts[elems[e].pri.v[2]].p;
            elems[e].p[3] = verts[elems[e].pri.v[3]].p;
            elems[e].p[4] = verts[elems[e].pri.v[4]].p;
            elems[e].p[5] = verts[elems[e].pri.v[5]].p;

            if(elems[e].pri.v[1] == vId) elems[e].n = 1;
            else if(elems[e].pri.v[2] == vId) elems[e].n = 2;
            else if(elems[e].pri.v[3] == vId) elems[e].n = 3;
            else if(elems[e].pri.v[4] == vId) elems[e].n = 4;
            else if(elems[e].pri.v[5] == vId) elems[e].n = 5;
            break;

        case HEX_ELEMENT_TYPE :
            elems[e].hex = hexs[elem.id];
            elems[e].p[0] = verts[elems[e].hex.v[0]].p;
            elems[e].p[1] = verts[elems[e].hex.v[1]].p;
            elems[e].p[2] = verts[elems[e].hex.v[2]].p;
            elems[e].p[3] = verts[elems[e].hex.v[3]].p;
            elems[e].p[4] = verts[elems[e].hex.v[4]].p;
            elems[e].p[5] = verts[elems[e].hex.v[5]].p;
            elems[e].p[6] = verts[elems[e].hex.v[6]].p;
            elems[e].p[7] = verts[elems[e].hex.v[7]].p;

            if(elems[e].hex.v[1] == vId) elems[e].n = 1;
            else if(elems[e].hex.v[2] == vId) elems[e].n = 2;
            else if(elems[e].hex.v[3] == vId) elems[e].n = 3;
            else if(elems[e].hex.v[4] == vId) elems[e].n = 4;
            else if(elems[e].hex.v[5] == vId) elems[e].n = 5;
            else if(elems[e].hex.v[6] == vId) elems[e].n = 6;
            else if(elems[e].hex.v[7] == vId) elems[e].n = 7;
            break;
        }
    }

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

        for(uint p=0; p < GRAD_SAMP_COUNT; ++p)
        {
            vec3 gradSamp = pos + GRAD_SAMPS[p] * nodeShift[nId];

            for(uint e=0, id = eBeg; id < eEnd; ++e, ++id)
            {
                elems[e].p[elems[e].n] = gradSamp;

                float qual = 0.0;
                switch(elems[e].type)
                {
                case TET_ELEMENT_TYPE :
                    qual = tetQuality(elems[e].p, elems[e].tet);
                    break;
                case PRI_ELEMENT_TYPE :
                    qual = priQuality(elems[e].p, elems[e].pri);
                    break;
                case HEX_ELEMENT_TYPE :
                    qual = hexQuality(elems[e].p, elems[e].hex);
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


        for(uint p=0; p < LINE_SAMP_COUNT; ++p)
        {
            vec3 lineSamp = pos + lineShift * LINE_SAMPS[p];

            for(uint e=0, id = eBeg; id < eEnd; ++e, ++id)
            {
                elems[e].p[elems[e].n] = lineSamp;

                float qual = 0.0;
                switch(elems[e].type)
                {
                case TET_ELEMENT_TYPE :
                    qual = tetQuality(elems[e].p, elems[e].tet);
                    break;
                case PRI_ELEMENT_TYPE :
                    qual = priQuality(elems[e].p, elems[e].pri);
                    break;
                case HEX_ELEMENT_TYPE :
                    qual = hexQuality(elems[e].p, elems[e].hex);
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
    uint vId = getInvocationVertexId();

    if(isSmoothableVertex(vId))
    {
        smoothVert(vId);
    }
}
