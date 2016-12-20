const uint POSITION_THREAD_COUNT = 8;
const uint NODE_THREAD_COUNT = 32;

const uint GRAD_SAMP_COUNT = 6;
const uint LINE_SAMP_COUNT = 8;

layout (local_size_x = POSITION_THREAD_COUNT, local_size_y = NODE_THREAD_COUNT, local_size_z = 1) in;

// Independent group range
uniform int GroupBase;
uniform int GroupSize;

uniform int SecurityCycleCount;
uniform float LocalSizeToNodeShift;

shared float nodeShift[NODE_THREAD_COUNT];
shared float patchQual[NODE_THREAD_COUNT][POSITION_THREAD_COUNT];


// Smoothing Helper
float computeLocalElementSize(in uint vId);
float tetQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Tet tet);
float priQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Pri pri);
float hexQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Hex hex);


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
    uint nId = gl_LocalInvocationID.y;

    Topo topo = topos[vId];
    uint neigElemCount = topo.neigElemCount;
    uint eBeg = topo.neigElemBase;
    uint eEnd = topo.neigElemBase + neigElemCount;

    if(pId == 0)
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

        float patchMin = 1.0;
        double patchMean = 0.0;

        if(pId < GRAD_SAMP_COUNT)
        {
            vec3 gradSamp = pos + GRAD_SAMPS[pId] * nodeShift[nId];

            for(uint e = eBeg; e < eEnd; ++e)
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
                    vertPos[elem.vId] = gradSamp;
                    qual = tetQuality(vertPos, tets[elem.id]);
                    break;

                case PRI_ELEMENT_TYPE :
                    vertPos[0] = verts[pris[elem.id].v[0]].p;
                    vertPos[1] = verts[pris[elem.id].v[1]].p;
                    vertPos[2] = verts[pris[elem.id].v[2]].p;
                    vertPos[3] = verts[pris[elem.id].v[3]].p;
                    vertPos[4] = verts[pris[elem.id].v[4]].p;
                    vertPos[5] = verts[pris[elem.id].v[5]].p;
                    vertPos[elem.vId] = gradSamp;
                    qual = priQuality(vertPos, pris[elem.id]);
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
                    vertPos[elem.vId] = gradSamp;
                    qual = hexQuality(vertPos, hexs[elem.id]);
                    break;
                }

                patchMin = min(patchMin, qual);
                patchMean += double(1.0 / qual);
            }

            if(patchMin <= 0.0)
                patchQual[nId][pId] = patchMin;
            else
                patchQual[nId][pId] = float(neigElemCount / patchMean);
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


        patchMin = 1.0;
        patchMean = 0.0;

        vec3 lineSamp = pos + lineShift * LINE_SAMPS[pId];

        for(uint e = eBeg; e < eEnd; ++e)
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
                vertPos[elem.vId] = lineSamp;
                qual = tetQuality(vertPos, tets[elem.id]);
                break;

            case PRI_ELEMENT_TYPE :
                vertPos[0] = verts[pris[elem.id].v[0]].p;
                vertPos[1] = verts[pris[elem.id].v[1]].p;
                vertPos[2] = verts[pris[elem.id].v[2]].p;
                vertPos[3] = verts[pris[elem.id].v[3]].p;
                vertPos[4] = verts[pris[elem.id].v[4]].p;
                vertPos[5] = verts[pris[elem.id].v[5]].p;
                vertPos[elem.vId] = lineSamp;
                qual = priQuality(vertPos, pris[elem.id]);
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
                vertPos[elem.vId] = lineSamp;
                qual = hexQuality(vertPos, hexs[elem.id]);
                break;
            }

            patchMin = min(patchMin, qual);
            patchMean += double(1.0 / qual);
        }

        if(patchMin <= 0.0)
            patchQual[nId][pId] = patchMin;
        else
            patchQual[nId][pId] = float(neigElemCount / patchMean);

        barrier();


        if(pId == 0)
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
    uint localId = gl_WorkGroupID.x * gl_WorkGroupSize.y + gl_LocalInvocationID.y;

    if(localId < GroupSize)
    {
        uint idx = GroupBase + localId;
        uint vId = groupMembers[idx];
        smoothVert(vId);
    }
}
