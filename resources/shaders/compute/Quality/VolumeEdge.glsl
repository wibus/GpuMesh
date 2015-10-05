float tetVolume(in vec3 vp[TET_VERTEX_COUNT]);
float priVolume(in vec3 vp[PRI_VERTEX_COUNT]);
float hexVolume(in vec3 vp[HEX_VERTEX_COUNT]);


float length2(in vec3 v)
{
    return dot(v, v);
}

float tetQuality(in vec3 vp[4])
{
    float volume = tetVolume(vp);

    float edge2Sum = 0.0;
    edge2Sum += length2(vp[0] - vp[1]);
    edge2Sum += length2(vp[0] - vp[2]);
    edge2Sum += length2(vp[0] - vp[3]);
    edge2Sum += length2(vp[1] - vp[2]);
    edge2Sum += length2(vp[2] - vp[3]);
    edge2Sum += length2(vp[3] - vp[1]);
    float edgeSum = sqrt(edge2Sum);

    return volume / (edgeSum*edgeSum*edgeSum)
            / 0.0080187537387448014348;
}

float priQuality(in vec3 vp[6])
{
    float volume = priVolume(vp);

    float edge2Sum = 0.0;
    edge2Sum += length2(vp[0] - vp[1]);
    edge2Sum += length2(vp[0] - vp[2]);
    edge2Sum += length2(vp[1] - vp[3]);
    edge2Sum += length2(vp[2] - vp[3]);
    edge2Sum += length2(vp[0] - vp[4]);
    edge2Sum += length2(vp[1] - vp[5]);
    edge2Sum += length2(vp[2] - vp[4]);
    edge2Sum += length2(vp[3] - vp[5]);
    edge2Sum += length2(vp[4] - vp[5]);
    float edgeSum = sqrt(edge2Sum);

    return min(volume / (edgeSum*edgeSum*edgeSum)
            / 0.016037507477489606339, 1.0);
}

float hexQuality(in vec3 vp[8])
{
    float volume = hexVolume(vp);

    float edge2Sum = 0.0;
    edge2Sum += length2(vp[0] - vp[1]);
    edge2Sum += length2(vp[0] - vp[2]);
    edge2Sum += length2(vp[1] - vp[3]);
    edge2Sum += length2(vp[2] - vp[3]);
    edge2Sum += length2(vp[0] - vp[4]);
    edge2Sum += length2(vp[1] - vp[5]);
    edge2Sum += length2(vp[2] - vp[6]);
    edge2Sum += length2(vp[3] - vp[7]);
    edge2Sum += length2(vp[4] - vp[5]);
    edge2Sum += length2(vp[4] - vp[6]);
    edge2Sum += length2(vp[5] - vp[7]);
    edge2Sum += length2(vp[6] - vp[7]);
    float edgeSum = sqrt(edge2Sum);

    return min(volume / (edgeSum*edgeSum*edgeSum)
            / 0.024056261216234407774, 1.0);
}
