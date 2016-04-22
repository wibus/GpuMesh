mat3 interpolateMetrics(in mat3 m1, in mat3 m2, float a)
{
    return mat3(
        mix(m1[0], m2[0], a),
        mix(m1[1], m2[1], a),
        mix(m1[2], m2[2], a));
}

mat3 vertMetric(in vec3 position)
{
    vec3 vp = position * vec3(7);

    float localElemSize = 0.0;
    localElemSize = 1.0 / pow(100, 1.0/3.0);

    float elemSize = localElemSize;
    float elemSizeInv2 = 1.0 / (elemSize * elemSize);

    float scale = pow(3, sin(vp.x));
    float targetElemSizeX = elemSize * scale;
    float targetElemSizeXInv2 = 1.0 / (targetElemSizeX * targetElemSizeX);
    float targetElemSizeZ = elemSize / scale;
    float targetElemSizeZInv2 = 1.0 / (targetElemSizeZ * targetElemSizeZ);

    float rx = targetElemSizeXInv2;
    float ry = elemSizeInv2;
    float rz = elemSizeInv2;

    return mat3(
        vec3(rx, 0,  0),
        vec3(0,  ry, 0),
        vec3(0,  0,  rz));
}

void boundingBox(out vec3 minBounds, out vec3 maxBounds)
{
    minBounds = vec3(1.0/0.0);
    maxBounds = vec3(-1.0/0.0);
    uint vertCount = verts.length();
    for(uint v=0; v < vertCount; ++v)
    {
        vec3 vertPos = verts[v].p;
        minBounds = min(minBounds, vertPos);
        maxBounds = max(maxBounds, vertPos);
    }
}

bool tetParams(in uint vi[4], in vec3 p, out float coor[4])
{
    dvec3 vp0 = dvec3(refVerts[vi[0]].p);
    dvec3 vp1 = dvec3(refVerts[vi[1]].p);
    dvec3 vp2 = dvec3(refVerts[vi[2]].p);
    dvec3 vp3 = dvec3(refVerts[vi[3]].p);

    dmat3 T = dmat3(vp0 - vp3, vp1 - vp3, vp2 - vp3);

    dvec3 y = inverse(T) * (dvec3(p) - vp3);
    coor[0] = float(y[0]);
    coor[1] = float(y[1]);
    coor[2] = float(y[2]);
    coor[3] = float(1.0LF - (y[0] + y[1] + y[2]));

    const float EPSILON_IN = -1e-8;
    bool isIn = (coor[0] >= EPSILON_IN && coor[1] >= EPSILON_IN &&
                 coor[2] >= EPSILON_IN && coor[3] >= EPSILON_IN);
    return isIn;
}
