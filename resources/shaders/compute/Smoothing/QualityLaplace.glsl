layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform float MoveCoeff;


float tetQuality(Tet tet);
float priQuality(Pri pri);
float hexQuality(Hex hex);
vec3 snapToBoundary(int boundaryID, vec3 pos);


void main()
{
    uint uid = gl_GlobalInvocationID.x;

    if(uid >= verts.length())
        return;

    Topo topo = topos[uid];
    uint neigElemCount = topo.neigElemCount;
    if(topo.type == TOPO_FIXED || neigElemCount == 0)
        return;


    // Compute patch center
    uint totalVertCount = 0;
    vec3 patchCenter = vec3(0.0);
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
    vec3 centerDist = patchCenter - pos;


    // Define propositions for new vertex's position
    const uint PROPOSITION_COUNT = 4;
    vec3 propositions[PROPOSITION_COUNT] = vec3[](
        pos,
        patchCenter - centerDist * MoveCoeff,
        patchCenter,
        patchCenter + centerDist * MoveCoeff
    );

    if(topo.type > 0)
        for(uint p=1; p < PROPOSITION_COUNT; ++p)
            propositions[p] = snapToBoundary(topo.type, propositions[p]);



    // Choose best position based on quality geometric mean
    uint bestProposition = 0;
    float bestQualityMean = 0.0;
    for(uint p=0; p < PROPOSITION_COUNT; ++p)
    {
        float qualityGeometricMean = 1.0;
        for(uint i=0, n = topo.neigElemBase; i < neigElemCount; ++i, ++n)
        {
            // Quality evaluation functions will use this updated position
            // to compute element shape measures.
            verts[uid].p = vec4(propositions[p], 0.0);

            NeigElem neigElem = neigElems[n];
            switch(neigElem.type)
            {
            case TET_ELEMENT_TYPE:
                qualityGeometricMean *= tetQuality(tets[neigElem.id]);
                break;

            case PRI_ELEMENT_TYPE:
                qualityGeometricMean *= priQuality(pris[neigElem.id]);
                break;

            case HEX_ELEMENT_TYPE:
                qualityGeometricMean *= hexQuality(hexs[neigElem.id]);
                break;
            }

            if(qualityGeometricMean <= 0.0)
            {
                qualityGeometricMean = 0.0;
                break;
            }
        }

        if(qualityGeometricMean > bestQualityMean)
        {
            bestQualityMean = qualityGeometricMean;
            bestProposition = p;
        }
    }


    // Update vertex's position
    verts[uid].p = vec4(propositions[bestProposition], 0.0);
}
