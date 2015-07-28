float length2(in vec3 v)
{
    return dot(v, v);
}

float tetQuality(in vec3 vp[4])
{
    float volume =  determinant(mat3(
        vp[TET_TETS[0].v[0]] - vp[TET_TETS[0].v[3]],
        vp[TET_TETS[0].v[1]] - vp[TET_TETS[0].v[3]],
        vp[TET_TETS[0].v[2]] - vp[TET_TETS[0].v[3]]));

    float edge2Sum = 0.0;
    edge2Sum += length2(vp[TET_EDGES[0].v[0]] - vp[TET_EDGES[0].v[1]]);
    edge2Sum += length2(vp[TET_EDGES[1].v[0]] - vp[TET_EDGES[1].v[1]]);
    edge2Sum += length2(vp[TET_EDGES[2].v[0]] - vp[TET_EDGES[2].v[1]]);
    edge2Sum += length2(vp[TET_EDGES[3].v[0]] - vp[TET_EDGES[3].v[1]]);
    edge2Sum += length2(vp[TET_EDGES[4].v[0]] - vp[TET_EDGES[4].v[1]]);
    edge2Sum += length2(vp[TET_EDGES[5].v[0]] - vp[TET_EDGES[5].v[1]]);
    float edgeSum = sqrt(edge2Sum);

    return min(volume / (edgeSum*edgeSum*edgeSum)
            / 0.048112522432468815548, 1.0); // Normalization constant
}

float priQuality(in vec3 vp[6])
{
    float volume = 0.0;
    volume += determinant(mat3(
        vp[PRI_TETS[0].v[0]] - vp[PRI_TETS[0].v[3]],
        vp[PRI_TETS[0].v[1]] - vp[PRI_TETS[0].v[3]],
        vp[PRI_TETS[0].v[2]] - vp[PRI_TETS[0].v[3]]));
    volume += determinant(mat3(
        vp[PRI_TETS[1].v[0]] - vp[PRI_TETS[1].v[3]],
        vp[PRI_TETS[1].v[1]] - vp[PRI_TETS[1].v[3]],
        vp[PRI_TETS[1].v[2]] - vp[PRI_TETS[1].v[3]]));
    volume += determinant(mat3(
        vp[PRI_TETS[2].v[0]] - vp[PRI_TETS[2].v[3]],
        vp[PRI_TETS[2].v[1]] - vp[PRI_TETS[2].v[3]],
        vp[PRI_TETS[2].v[2]] - vp[PRI_TETS[2].v[3]]));

    float edge2Sum = 0.0;
    edge2Sum += length2(vp[PRI_EDGES[0].v[0]] - vp[PRI_EDGES[0].v[1]]);
    edge2Sum += length2(vp[PRI_EDGES[1].v[0]] - vp[PRI_EDGES[1].v[1]]);
    edge2Sum += length2(vp[PRI_EDGES[2].v[0]] - vp[PRI_EDGES[2].v[1]]);
    edge2Sum += length2(vp[PRI_EDGES[3].v[0]] - vp[PRI_EDGES[3].v[1]]);
    edge2Sum += length2(vp[PRI_EDGES[4].v[0]] - vp[PRI_EDGES[4].v[1]]);
    edge2Sum += length2(vp[PRI_EDGES[5].v[0]] - vp[PRI_EDGES[5].v[1]]);
    edge2Sum += length2(vp[PRI_EDGES[6].v[0]] - vp[PRI_EDGES[6].v[1]]);
    edge2Sum += length2(vp[PRI_EDGES[7].v[0]] - vp[PRI_EDGES[7].v[1]]);
    edge2Sum += length2(vp[PRI_EDGES[8].v[0]] - vp[PRI_EDGES[8].v[1]]);
    float edgeSum = sqrt(edge2Sum);

    return min(volume / (edgeSum*edgeSum*edgeSum)
            / 0.096225044864937631095, 1.0); // Normalization constant
}

float hexQuality(in vec3 vp[8])
{
    float volume = 0.0;
    volume += determinant(mat3(
        vp[HEX_TETS[0].v[0]] - vp[HEX_TETS[0].v[3]],
        vp[HEX_TETS[0].v[1]] - vp[HEX_TETS[0].v[3]],
        vp[HEX_TETS[0].v[2]] - vp[HEX_TETS[0].v[3]]));
    volume += determinant(mat3(
        vp[HEX_TETS[1].v[0]] - vp[HEX_TETS[1].v[3]],
        vp[HEX_TETS[1].v[1]] - vp[HEX_TETS[1].v[3]],
        vp[HEX_TETS[1].v[2]] - vp[HEX_TETS[1].v[3]]));
    volume += determinant(mat3(
        vp[HEX_TETS[2].v[0]] - vp[HEX_TETS[2].v[3]],
        vp[HEX_TETS[2].v[1]] - vp[HEX_TETS[2].v[3]],
        vp[HEX_TETS[2].v[2]] - vp[HEX_TETS[2].v[3]]));
    volume += determinant(mat3(
        vp[HEX_TETS[3].v[0]] - vp[HEX_TETS[3].v[3]],
        vp[HEX_TETS[3].v[1]] - vp[HEX_TETS[3].v[3]],
        vp[HEX_TETS[3].v[2]] - vp[HEX_TETS[3].v[3]]));
    volume += determinant(mat3(
        vp[HEX_TETS[4].v[0]] - vp[HEX_TETS[4].v[3]],
        vp[HEX_TETS[4].v[1]] - vp[HEX_TETS[4].v[3]],
        vp[HEX_TETS[4].v[2]] - vp[HEX_TETS[4].v[3]]));

    float edge2Sum = 0.0;
    edge2Sum += length2(vp[HEX_EDGES[0].v[0]] - vp[HEX_EDGES[0].v[1]]);
    edge2Sum += length2(vp[HEX_EDGES[1].v[0]] - vp[HEX_EDGES[1].v[1]]);
    edge2Sum += length2(vp[HEX_EDGES[2].v[0]] - vp[HEX_EDGES[2].v[1]]);
    edge2Sum += length2(vp[HEX_EDGES[3].v[0]] - vp[HEX_EDGES[3].v[1]]);
    edge2Sum += length2(vp[HEX_EDGES[4].v[0]] - vp[HEX_EDGES[4].v[1]]);
    edge2Sum += length2(vp[HEX_EDGES[5].v[0]] - vp[HEX_EDGES[5].v[1]]);
    edge2Sum += length2(vp[HEX_EDGES[6].v[0]] - vp[HEX_EDGES[6].v[1]]);
    edge2Sum += length2(vp[HEX_EDGES[7].v[0]] - vp[HEX_EDGES[7].v[1]]);
    edge2Sum += length2(vp[HEX_EDGES[8].v[0]] - vp[HEX_EDGES[8].v[1]]);
    edge2Sum += length2(vp[HEX_EDGES[9].v[0]] - vp[HEX_EDGES[9].v[1]]);
    edge2Sum += length2(vp[HEX_EDGES[10].v[0]] - vp[HEX_EDGES[10].v[1]]);
    edge2Sum += length2(vp[HEX_EDGES[11].v[0]] - vp[HEX_EDGES[11].v[1]]);
    float edgeSum = sqrt(edge2Sum);

    return min(volume / (edgeSum*edgeSum*edgeSum)
            / 0.14433756729740643276, 1.0); // Normalization constant
}
