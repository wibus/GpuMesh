const uint SPAWN_COUNT = 64;
const uint ELEM_SLOT_COUNT = 128;

layout (local_size_x = SPAWN_COUNT, local_size_y = 1, local_size_z = 1) in;


layout(shared, binding = SPAWN_OFFSETS_BUFFER_BINDING) buffer Offsets
{
    vec4 offsets[];
};

struct TetVert
{
    vec3 p[TET_VERTEX_COUNT];
};

shared Tet tetElems[ELEM_SLOT_COUNT];
shared TetVert tetVerts[ELEM_SLOT_COUNT];
shared float qualities[SPAWN_COUNT];
uniform float MoveCoeff;

// Independent group range
uniform int GroupBase;
uniform int GroupSize;


// Externally defined
float tetQuality(in vec3 vp[TET_VERTEX_COUNT], inout Tet tet);
float finalizePatchQuality(in double patchQuality, in double patchWeight);
void accumulatePatchQuality(
        inout double patchQuality,
        inout double patchWeight,
        in double elemQuality);

// Smoothing helper
float computeLocalElementSize(in uint vId);


// ENTRY POINT //
void smoothVert(uint vId)
{
    uint lId = gl_LocalInvocationIndex;
    Topo topo = topos[vId];

    uint neigElemCount = topo.neigElemCount;
    uint firstLoad = (neigElemCount * lId) / gl_WorkGroupSize.x;
    uint lastLoad = (neigElemCount * (lId+1)) / gl_WorkGroupSize.x;

    for(uint eId = firstLoad; eId < lastLoad; ++eId)
    {
        NeigElem elem = neigElems[topo.neigElemBase + eId];

        Tet tet = tets[elem.id];
        tetElems[eId] = tet;

        tetVerts[eId] = TetVert(vec3[](
            verts[tet.v[0]].p,
            verts[tet.v[1]].p,
            verts[tet.v[2]].p,
            verts[tet.v[3]].p
        ));
    }

    barrier();

    // Compute local element size
    float localSize = computeLocalElementSize(vId);
    float scale = localSize * MoveCoeff;

    vec4 offset = offsets[lId];

    for(int iter=0; iter < 2; ++iter)
    {
        vec3 spawnPos = verts[vId].p + vec3(offset) * scale;


        double patchWeight = 0.0;
        double patchQuality = 1.0;
        for(uint i=0; i < neigElemCount; ++i)
        {
            Tet tetElem = tetElems[i];
            TetVert tetVert = tetVerts[i];

            if(tetElem.v[0] == vId)
                tetVert.p[0] = spawnPos;
            else if(tetElem.v[1] == vId)
                tetVert.p[1] = spawnPos;
            else if(tetElem.v[2] == vId)
                tetVert.p[2] = spawnPos;
            else if(tetElem.v[3] == vId)
                tetVert.p[3] = spawnPos;

            accumulatePatchQuality(
                patchQuality, patchWeight,
                double(tetQuality(tetVert.p, tetElem)));
        }

        qualities[lId] = finalizePatchQuality(patchQuality, patchWeight);

        barrier();

        if(lId == 0)
        {
            uint bestLoc = 0;
            float bestQual = -1.0/0.0; // -Inf

            for(int i=0; i < SPAWN_COUNT; ++i)
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

        barrier();

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
