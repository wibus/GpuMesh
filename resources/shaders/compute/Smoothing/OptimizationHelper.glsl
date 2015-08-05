float tetQuality(in vec3 vp[TET_VERTEX_COUNT]);
float priQuality(in vec3 vp[PRI_VERTEX_COUNT]);
float hexQuality(in vec3 vp[HEX_VERTEX_COUNT]);


const uint MAX_PROPOSITION_COUNT = 4;


vec3 findPatchCenter(in uint v)
{
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

    vec3 pos = vec3(verts[uid].p);
    patchCenter = (patchCenter - pos * float(neigElemCount))
                    / float(totalVertCount);

    return patchCenter;
}


void integrateQuality(inout float total, float shape)
{
    total *= shape;
}

void testTetPropositions(
        uint vertId,
        Tet elem,
        in vec3 propositions[MAX_PROPOSITION_COUNT],
        inout float propQualities[MAX_PROPOSITION_COUNT],
        in uint propositionCount)
{
    // Extract element's vertices
    vec3 vp[TET_VERTEX_COUNT] = vec3[](
        verts[elem.v[0]].p,
        verts[elem.v[1]].p,
        verts[elem.v[2]].p,
        verts[elem.v[3]].p
    );

    // Find Vertex position in element
    uint elemVertId = 0;
    if(vertId == elem.v[1])
        elemVertId = 1;
    else if(vertId == elem.v[2])
        elemVertId = 2;
    else if(vertId == elem.v[3])
        elemVertId = 3;

    for(uint p=0; p < propositionCount; ++p)
    {
        if(propQualities[p] > 0.0)
        {
            vp[elemVertId] = propositions[p];

            integrateQuality(
                propQualities[p],
                tetQuality(vp));
        }
    }
}

void testPriPropositions(
        uint vertId,
        Pri elem,
        in vec3 propositions[MAX_PROPOSITION_COUNT],
        inout float propQualities[MAX_PROPOSITION_COUNT],
        in uint propositionCount)
{
    // Extract element's vertices
    vec3 vp[PRI_VERTEX_COUNT] = vec3[](
        verts[elem.v[0]].p,
        verts[elem.v[1]].p,
        verts[elem.v[2]].p,
        verts[elem.v[3]].p,
        verts[elem.v[4]].p,
        verts[elem.v[5]].p
    );

    // Find Vertex position in element
    uint elemVertId = 0;
    if(vertId == elem.v[1])
        elemVertId = 1;
    else if(vertId == elem.v[2])
        elemVertId = 2;
    else if(vertId == elem.v[3])
        elemVertId = 3;
    else if(vertId == elem.v[4])
        elemVertId = 4;
    else if(vertId == elem.v[5])
        elemVertId = 5;

    for(uint p=0; p < propositionCount; ++p)
    {
        if(propQualities[p] > 0.0)
        {
            vp[elemVertId] = propositions[p];

            integrateQuality(
                propQualities[p],
                priQuality(vp));
        }
    }
}

void testHexPropositions(
        uint vertId,
        Hex elem,
        in vec3 propositions[MAX_PROPOSITION_COUNT],
        inout float propQualities[MAX_PROPOSITION_COUNT],
        in uint propositionCount)
{
    // Extract element's vertices
    vec3 vp[HEX_VERTEX_COUNT] = vec3[](
        verts[elem.v[0]].p,
        verts[elem.v[1]].p,
        verts[elem.v[2]].p,
        verts[elem.v[3]].p,
        verts[elem.v[4]].p,
        verts[elem.v[5]].p,
        verts[elem.v[6]].p,
        verts[elem.v[7]].p
    );

    // Find vertex position in element
    uint elemVertId = 0;
    if(vertId == elem.v[1])
        elemVertId = 1;
    else if(vertId == elem.v[2])
        elemVertId = 2;
    else if(vertId == elem.v[3])
        elemVertId = 3;
    else if(vertId == elem.v[4])
        elemVertId = 4;
    else if(vertId == elem.v[5])
        elemVertId = 5;
    else if(vertId == elem.v[6])
        elemVertId = 6;
    else if(vertId == elem.v[7])
        elemVertId = 7;

    for(uint p=0; p < propositionCount; ++p)
    {
        if(propQualities[p] > 0.0)
        {
            vp[elemVertId] = propositions[p];

            integrateQuality(
                propQualities[p],
                hexQuality(vp));
        }
    }
}
