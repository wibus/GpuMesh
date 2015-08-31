// Element quality interface
float tetQuality(in Tet tet);
float priQuality(in Pri pri);
float hexQuality(in Hex hex);


// Worgroup invocation disptach mode
uniform int DispatchMode = 0;
const int DISPATCH_MODE_CLUSTER = 0;
const int DISPATCH_MODE_SCATTER = 1;

uint getInvocationVertexId()
{
    uint vId;

    switch(DispatchMode)
    {
    case DISPATCH_MODE_SCATTER :
        vId = gl_LocalInvocationID.x * gl_NumWorkGroups.x + gl_WorkGroupID.x;
        break;

    case DISPATCH_MODE_CLUSTER :
        // FALLTHROUGH default case
    default :
        vId = gl_GlobalInvocationID.x;
        break;
    }

    return vId;
}

uint getInvocationTetId()
{
    uint eId;

    // No scatter mode available
    eId = gl_GlobalInvocationID.x;

    return eId;
}

uint getInvocationPriId()
{
    uint eId;

    // No scatter mode available
    eId = gl_GlobalInvocationID.x;

    return eId;
}


uint getInvocationHexId()
{
    uint eId;

    // No scatter mode available
    eId = gl_GlobalInvocationID.x;

    return eId;
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
    patchQuality = min(patchQuality * elemQuality, elemQuality);
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
