const uint NODE_THREAD_COUNT = 32;
const uint ELEMENT_THREAD_COUNT = 8;

const int MIN_MAX = 2147483647;


// Independent group range
uniform int GroupBase;
uniform int GroupSize;

uniform float GainThreshold;
uniform int SecurityCycleCount;
uniform float LocalSizeToNodeShift;
uniform float Alpha;
uniform float Beta;
uniform float Gamma;
uniform float Delta;

shared int patchMin[NODE_THREAD_COUNT];
shared float patchMean[NODE_THREAD_COUNT];


// Smoothing Helper
uint getInvocationVertexId();
bool isSmoothableVertex(uint vId);
float computeLocalElementSize(in uint vId);
float tetQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Tet tet);
float priQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Pri pri);
float hexQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Hex hex);
void swap(inout vec4 v1, inout vec4 v2);


subroutine float parallelPatchQualitySub(
        in uint nBeg, in uint nEnd,
        in uint neigElemCount,
        in vec3 pos);
layout(location=PARALLEL_PATCH_QUALITY_SUBROUTINE_LOC)
subroutine uniform parallelPatchQualitySub parallelPatchQualityUni;

float parallelPatchQuality(
        in uint nBeg, in uint nEnd,
        in uint neigElemCount,
        in vec3 pos)
{
    return parallelPatchQualityUni(
                nBeg, nEnd,
                neigElemCount,
                pos);
}


layout(index=PARALLEL_PATCH_QUALITY_SUBROUTINE_IDX) subroutine(parallelPatchQualitySub)
float parallelPatchQualityImpl(
        in uint nBeg, in uint nEnd,
        in uint neigElemCount,
        in vec3 pos)
{
    uint nId = gl_LocalInvocationID.y;

    patchMin[nId] = MIN_MAX;
    patchMean[nId] = 0.0;

    barrier();


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
            vertPos[elem.vId] = pos;
            qual = tetQuality(vertPos, tets[elem.id]);
            break;

        case PRI_ELEMENT_TYPE :
            vertPos[0] = verts[pris[elem.id].v[0]].p;
            vertPos[1] = verts[pris[elem.id].v[1]].p;
            vertPos[2] = verts[pris[elem.id].v[2]].p;
            vertPos[3] = verts[pris[elem.id].v[3]].p;
            vertPos[4] = verts[pris[elem.id].v[4]].p;
            vertPos[5] = verts[pris[elem.id].v[5]].p;
            vertPos[elem.vId] = pos;
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
            vertPos[elem.vId] = pos;
            qual = hexQuality(vertPos, hexs[elem.id]);
            break;
        }

        atomicMin(patchMin[nId], int(qual * MIN_MAX));
        atomicAdd(patchMean[nId], 1.0 / qual);
    }

    barrier();


    float patchQual = 0.0;

    if(patchMin[nId] <= 0.0)
        patchQual = patchMin[nId] / float(MIN_MAX);
    else
        patchQual = neigElemCount / patchMean[nId];

    return patchQual;
}



// ENTRY POINT //
void smoothVert(uint vId)
{
    uint eId = gl_LocalInvocationID.x;

    Topo topo = topos[vId];
    uint neigElemCount = topo.neigElemCount;
    uint nBeg = topo.neigElemBase + (eId * neigElemCount) / ELEMENT_THREAD_COUNT;
    uint nEnd = topo.neigElemBase + ((eId+1) * neigElemCount) / ELEMENT_THREAD_COUNT;


    // Compute local element size
    float localSize = computeLocalElementSize(vId);

    // Initialize node shift distance
    float nodeShift = localSize * LocalSizeToNodeShift;

    vec3 pos = verts[vId].p;
    vec4 vo = vec4(pos, parallelPatchQuality(nBeg, nEnd, neigElemCount, pos));

    vec4 simplex[TET_VERTEX_COUNT] = {
        vec4(pos + vec3(nodeShift, 0, 0), 0),
        vec4(pos + vec3(0, nodeShift, 0), 0),
        vec4(pos + vec3(0, 0, nodeShift), 0),
        vo
    };

    int cycle = 0;
    bool reset = false;
    bool terminated = false;
    while(!terminated)
    {
        for(uint p=0; p < TET_VERTEX_COUNT-1; ++p)
        {
            // Since 'pos' is a reference on vertex's position
            // modifing its value here should be seen by the evaluator
            verts[vId].p = vec3(simplex[p]);

            // Compute patch quality
            simplex[p].w = parallelPatchQuality(nBeg, nEnd, neigElemCount, verts[vId].p);
        }

        // Mini bubble sort
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);
        if(simplex[1].w > simplex[2].w)
            swap(simplex[1], simplex[2]);
        if(simplex[2].w > simplex[3].w)
            swap(simplex[2], simplex[3]);
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);
        if(simplex[1].w > simplex[2].w)
            swap(simplex[1], simplex[2]);
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);


        for(; cycle < SecurityCycleCount; ++cycle)
        {
            // Centroid
            vec3 c = 1/3.0f * (
                vec3(simplex[1]) +
                vec3(simplex[2]) +
                vec3(simplex[3]));

            float f = 0.0;

            // Reflect
            verts[vId].p = c + Alpha*(c - vec3(simplex[0]));
            float fr = f = parallelPatchQuality(nBeg, nEnd, neigElemCount, verts[vId].p);

            vec3 xr = verts[vId].p;

            // Expand
            if(simplex[3].w < fr)
            {
                verts[vId].p = c + Gamma*(verts[vId].p - c);
                float fe = f = parallelPatchQuality(nBeg, nEnd, neigElemCount, verts[vId].p);

                if(fe <= fr)
                {
                    verts[vId].p = xr;
                    f = fr;
                }
            }
            // Contract
            else if(simplex[1].w >= fr)
            {
                // Outside
                if(fr > simplex[0].w)
                {
                    verts[vId].p = c + Beta*(vec3(xr) - c);
                    f = parallelPatchQuality(nBeg, nEnd, neigElemCount, verts[vId].p);
                }
                // Inside
                else
                {
                    verts[vId].p = c + Beta*(vec3(simplex[0]) - c);
                    f = parallelPatchQuality(nBeg, nEnd, neigElemCount, verts[vId].p);
                }
            }

            // Insert new vertex in the working simplex
            vec4 vertex = vec4(verts[vId].p, f);
            if(vertex.w > simplex[3].w)
                swap(simplex[3], vertex);
            if(vertex.w > simplex[2].w)
                swap(simplex[2], vertex);
            if(vertex.w > simplex[1].w)
                swap(simplex[1], vertex);
            if(vertex.w > simplex[0].w)
                swap(simplex[0], vertex);


            if( (simplex[3].w - simplex[1].w) < GainThreshold )
            {
                terminated = true;
                break;
            }
        }

        if( terminated || (cycle >= SecurityCycleCount && reset) )
        {
            break;
        }
        else
        {
            simplex[0] = vo - vec4(nodeShift, 0, 0, 0);
            simplex[1] = vo - vec4(0, nodeShift, 0, 0);
            simplex[2] = vo - vec4(0, 0, nodeShift, 0);
            simplex[3] = vo;
            reset = true;
            cycle = 0;
        }
    }

    verts[vId].p = vec3(simplex[3]);
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
