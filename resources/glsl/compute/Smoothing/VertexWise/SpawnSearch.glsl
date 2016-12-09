const uint SPAWN_COUNT = 64;
const uint ELEM_SLOT_COUNT = 128;

layout (local_size_x = SPAWN_COUNT, local_size_y = 1, local_size_z = 1) in;

layout(shared, binding = SPAWN_OFFSETS_BUFFER_BINDING) buffer Offsets
{
    vec4 offsets[];
};

shared PatchElem patchElems[ELEM_SLOT_COUNT];
shared float qualities[SPAWN_COUNT];
uniform float MoveCoeff;


// Independent group range
uniform int GroupBase;
uniform int GroupSize;


// Smoothing helper
float computeLocalElementSize(in uint vId);
float tetQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Tet tet);
float priQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Pri pri);
float hexQuality(in vec3 vp[PARAM_VERTEX_COUNT], inout Hex hex);
float finalizePatchQuality(in double patchQuality, in double patchWeight);
void accumulatePatchQuality(
        inout double patchQuality,
        inout double patchWeight,
        in double elemQuality);


// ENTRY POINT //
void smoothVert(uint vId)
{
    uint lId = gl_LocalInvocationID.x;

    Topo topo = topos[vId];
    uint neigElemCount = topo.neigElemCount;
    uint firstLoad = (neigElemCount * lId) / gl_WorkGroupSize.x;
    uint lastLoad = (neigElemCount * (lId+1)) / gl_WorkGroupSize.x;

    for(uint e = firstLoad; e < lastLoad; ++e)
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

    // Compute local element size
    float localSize = computeLocalElementSize(vId);
    float scale = localSize * MoveCoeff;

    vec4 offset = offsets[lId];

    for(int iter=0; iter < 2; ++iter)
    {
        barrier();

        vec3 spawnPos = verts[vId].p + vec3(offset) * scale;


        double patchWeight = 0.0;
        double patchQuality = 0.0;
        for(uint i=0; i < neigElemCount; ++i)
        {
            vec3 vertPos[] = vec3[](
                patchElems[i].p[0],
                patchElems[i].p[1],
                patchElems[i].p[2],
                patchElems[i].p[3],
                patchElems[i].p[4],
                patchElems[i].p[5],
                patchElems[i].p[6],
                patchElems[i].p[7]
            );

            vertPos[patchElems[i].n] = spawnPos;

            float qual = 0.0;
            switch(patchElems[i].type)
            {
            case TET_ELEMENT_TYPE :
                qual = tetQuality(vertPos, patchElems[i].tet);
                break;
            case PRI_ELEMENT_TYPE :
                qual = priQuality(vertPos, patchElems[i].pri);
                break;
            case HEX_ELEMENT_TYPE :
                qual = hexQuality(vertPos, patchElems[i].hex);
                break;
            }

            accumulatePatchQuality(
                patchQuality, patchWeight,
                double(qual));
        }

        qualities[lId] = finalizePatchQuality(patchQuality, patchWeight);

        barrier();

        if(lId == 0)
        {
            uint bestLoc = 0;
            float bestQual = qualities[0];

            for(int i=1; i < SPAWN_COUNT; ++i)
            {
                if(qualities[i] > bestQual)
                {
                    bestLoc = i;
                    bestQual = qualities[i];
                }
            }

            // Update vertex's position
            verts[vId].p += vec3(offsets[bestLoc]) * scale;
        }

        scale /= 3.0;
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
