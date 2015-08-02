float length2(in vec3 v)
{
    return dot(v, v);
}

float tetQuality(in vec3 vp[4])
{
    float volume =  determinant(mat3(
        vp[0] - vp[3],
        vp[1] - vp[3],
        vp[2] - vp[3]));

    float edge2Sum = 0.0;
    edge2Sum += length2(vp[0] - vp[1]);
    edge2Sum += length2(vp[0] - vp[2]);
    edge2Sum += length2(vp[0] - vp[3]);
    edge2Sum += length2(vp[1] - vp[2]);
    edge2Sum += length2(vp[2] - vp[3]);
    edge2Sum += length2(vp[3] - vp[1]);
    float edgeSum = sqrt(edge2Sum);

    return min(volume / (edgeSum*edgeSum*edgeSum)
            / 0.048112522432468815548, 1.0); // Normalization constant
}

float priQuality(in vec3 vp[6])
{
    float volume = 0.0;
    volume += determinant(mat3(
        vp[4] - vp[2],
        vp[0] - vp[2],
        vp[1] - vp[2]));
    volume += determinant(mat3(
        vp[5] - vp[2],
        vp[1] - vp[2],
        vp[3] - vp[2]));
    volume += determinant(mat3(
        vp[4] - vp[2],
        vp[1] - vp[2],
        vp[5] - vp[2]));

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
            / 0.096225044864937631095, 1.0); // Normalization constant
}

float hexQuality(in vec3 vp[8])
{
    float volume = 0.0;
    volume += determinant(mat3(
        vp[0] - vp[2],
        vp[1] - vp[2],
        vp[4] - vp[2]));
    volume += determinant(mat3(
        vp[3] - vp[1],
        vp[2] - vp[1],
        vp[7] - vp[1]));
    volume += determinant(mat3(
        vp[5] - vp[4],
        vp[1] - vp[4],
        vp[7] - vp[4]));
    volume += determinant(mat3(
        vp[6] - vp[7],
        vp[2] - vp[7],
        vp[4] - vp[7]));
    volume += determinant(mat3(
        vp[1] - vp[2],
        vp[7] - vp[2],
        vp[4] - vp[2]));

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
            / 0.14433756729740643276, 1.0); // Normalization constant
}
