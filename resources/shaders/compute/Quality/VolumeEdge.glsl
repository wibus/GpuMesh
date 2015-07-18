
float tetQuality(Tet tet)
{
    vec3 ev[] = {
        vec3(verts[tet.v[0]].p),
        vec3(verts[tet.v[1]].p),
        vec3(verts[tet.v[2]].p),
        vec3(verts[tet.v[3]].p),
    };

    float volume = 0.0;
    for(int t=0; t < TET_TET_COUNT; ++t)
    {
        volume += determinant(mat3(
            ev[TET_TETS[t].v[0]] - ev[TET_TETS[t].v[3]],
            ev[TET_TETS[t].v[1]] - ev[TET_TETS[t].v[3]],
            ev[TET_TETS[t].v[2]] - ev[TET_TETS[t].v[3]]));
    }

    float edge2Sum = 0.0;
    for(int e=0; e < TET_EDGE_COUNT; ++e)
    {
        vec3 edge = ev[TET_EDGES[e].v[0]] -
                    ev[TET_EDGES[e].v[1]];
        edge2Sum += dot(edge, edge);
    }
    float edgeSum = sqrt(edge2Sum);

    return min(volume / (edgeSum*edgeSum*edgeSum)
            / 0.048112522432468815548, 1.0); // Normalization constant
}

float priQuality(Pri pri)
{
    vec3 ev[] = {
        vec3(verts[pri.v[0]].p),
        vec3(verts[pri.v[1]].p),
        vec3(verts[pri.v[2]].p),
        vec3(verts[pri.v[3]].p),
        vec3(verts[pri.v[4]].p),
        vec3(verts[pri.v[5]].p)
    };

    float volume = 0.0;
    for(int t=0; t < PRI_TET_COUNT; ++t)
    {
        volume += determinant(mat3(
            ev[PRI_TETS[t].v[0]] - ev[PRI_TETS[t].v[3]],
            ev[PRI_TETS[t].v[1]] - ev[PRI_TETS[t].v[3]],
            ev[PRI_TETS[t].v[2]] - ev[PRI_TETS[t].v[3]]));
    }

    float edge2Sum = 0.0;
    for(int e=0; e < PRI_EDGE_COUNT; ++e)
    {
        vec3 edge = ev[PRI_EDGES[e].v[0]] -
                    ev[PRI_EDGES[e].v[1]];
        edge2Sum += dot(edge, edge);
    }
    float edgeSum = sqrt(edge2Sum);

    return min(volume / (edgeSum*edgeSum*edgeSum)
            / 0.088228615568855695006, 1.0); // Normalization constant
}

float hexQuality(Hex hex)
{
    vec3 ev[] = {
        vec3(verts[hex.v[0]].p),
        vec3(verts[hex.v[1]].p),
        vec3(verts[hex.v[2]].p),
        vec3(verts[hex.v[3]].p),
        vec3(verts[hex.v[4]].p),
        vec3(verts[hex.v[5]].p),
        vec3(verts[hex.v[6]].p),
        vec3(verts[hex.v[7]].p)
    };

    float volume = 0.0;
    for(int t=0; t < HEX_TET_COUNT; ++t)
    {
        volume += determinant(mat3(
            ev[HEX_TETS[t].v[0]] - ev[HEX_TETS[t].v[3]],
            ev[HEX_TETS[t].v[1]] - ev[HEX_TETS[t].v[3]],
            ev[HEX_TETS[t].v[2]] - ev[HEX_TETS[t].v[3]]));
    }

    float edge2Sum = 0.0;
    for(int e=0; e < HEX_EDGE_COUNT; ++e)
    {
        vec3 edge = ev[HEX_EDGES[e].v[0]] -
                    ev[HEX_EDGES[e].v[1]];
        edge2Sum += dot(edge, edge);
    }
    float edgeSum = sqrt(edge2Sum);

    return min(volume / (edgeSum*edgeSum*edgeSum)
            / 0.14433756729740643276, 1.0); // Normalization constant
}
