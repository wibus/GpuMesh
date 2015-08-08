float tetQuality(in Tet tet);
float priQuality(in Pri pri);
float hexQuality(in Hex hex);

const uint MAX_PROPOSITION_COUNT = 4;


vec3 findPatchCenter(in uint v, in Topo topo)
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

    vec3 pos = vec3(verts[v].p);
    patchCenter = (patchCenter - pos * float(neigElemCount))
                    / float(totalVertCount);
    return patchCenter;
}


void accumulatePatchQuality(in float elemQ, inout float patchQ)
{
    patchQ *= elemQ;
}

void finalizePatchQuality(inout float patchQ)
{
}
