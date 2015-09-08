// Patch exclusive group range
uniform int GroupBase;
uniform int GroupSize;

// Element quality interface
float tetQuality(in Tet tet);
float priQuality(in Pri pri);
float hexQuality(in Hex hex);

uint getInvocationVertexId()
{
    // Default value is invalid
    // See isSmoothableVertex()
    uint vId = verts.length();

    // Assign real index only if this
    // invocation does not overflow
    if(gl_GlobalInvocationID.x < GroupSize)
        vId = groupMembers[GroupBase + gl_GlobalInvocationID.x];

    return vId;
}

uint getInvocationTetId()
{
    return gl_GlobalInvocationID.x;
}

uint getInvocationPriId()
{
    return gl_GlobalInvocationID.x;
}

uint getInvocationHexId()
{
    return gl_GlobalInvocationID.x;
}


bool isSmoothableVertex(uint vId)
{
    if(vId >= verts.length())
        return false;

    Topo topo = topos[vId];
    if(topo.type == TOPO_FIXED)
        return false;

    if(topo.neigElemCount == 0)
        return false;

    return true;
}

bool isSmoothableTet(uint eId)
{
    if(eId >= tets.length())
        return false;

    return true;
}

bool isSmoothablePri(uint eId)
{
    if(eId >= pris.length())
        return false;

    return true;
}

bool isSmoothableHex(uint eId)
{
    if(eId >= hexs.length())
        return false;

    return true;
}


float computeLocalElementSize(in uint vId)
{
    vec3 pos = vec3(verts[vId].p);
    Topo topo = topos[vId];

    float totalSize = 0.0;
    uint neigVertCount = topo.neigVertCount;
    for(uint i=0, n = topo.neigVertBase; i < neigVertCount; ++i, ++n)
    {
        totalSize += length(pos - vec3(verts[neigVerts[n].v].p));
    }

    return totalSize / neigVertCount;
}

vec3 computePatchCenter(in uint vId)
{
    Topo topo = topos[vId];

    uint totalVertCount = 0;
    vec3 patchCenter = vec3(0.0);
    uint neigElemCount = topo.neigElemCount;
    for(uint i=0, n = topo.neigElemBase; i<neigElemCount; ++i, ++n)
    {
        NeigElem neigElem = neigElems[n];
        switch(neigElem.type)
        {
        case TET_ELEMENT_TYPE:
            totalVertCount += TET_VERTEX_COUNT - 1;
            for(uint i=0; i < TET_VERTEX_COUNT; ++i)
                patchCenter += vec3(verts[tets[neigElem.id].v[i]].p);
            break;

        case PRI_ELEMENT_TYPE:
            totalVertCount += PRI_VERTEX_COUNT - 1;
            for(uint i=0; i < PRI_VERTEX_COUNT; ++i)
                patchCenter += vec3(verts[pris[neigElem.id].v[i]].p);
            break;

        case HEX_ELEMENT_TYPE:
            totalVertCount += HEX_VERTEX_COUNT - 1;
            for(uint i=0; i < HEX_VERTEX_COUNT; ++i)
                patchCenter += vec3(verts[hexs[neigElem.id].v[i]].p);
            break;
        }
    }

    vec3 pos = vec3(verts[vId].p);
    patchCenter = (patchCenter - pos * float(neigElemCount))
                    / float(totalVertCount);
    return patchCenter;
}


void accumulatePatchQuality(
        inout float patchQuality,
        inout float patchWeight,
        in float elemQuality)
{
    patchQuality = min(
        min(patchQuality, elemQuality),  // If sign(patch) != sign(elem)
        min(patchQuality * elemQuality,  // If sign(patch) & sign(elem) > 0
            patchQuality + elemQuality));// If sign(patch) & sign(elem) < 0
}

float finalizePatchQuality(in float patchQuality, in float patchWeight)
{
    return patchQuality;
}

float computePatchQuality(in uint vId)
{
    Topo topo = topos[vId];

    float patchWeight = 0.0;
    float patchQuality = 1.0;
    uint neigElemCount = topo.neigElemCount;
    for(uint i=0, n = topo.neigElemBase; i < neigElemCount; ++i, ++n)
    {
        NeigElem neigElem = neigElems[n];
        switch(neigElem.type)
        {
        case TET_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                tetQuality(tets[neigElem.id]));
            break;

        case PRI_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                priQuality(pris[neigElem.id]));
            break;

        case HEX_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                hexQuality(hexs[neigElem.id]));
            break;
        }
    }

    return finalizePatchQuality(patchQuality, patchWeight);
}
