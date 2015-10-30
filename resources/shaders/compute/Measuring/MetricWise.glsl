mat3 metricAt(in vec3 pos);

// Distances
float riemannianDistance(in vec3 a, in vec3 b)
{
    vec3 abDiff = b - a;
    vec3 middle = (a + b) / 2.0;
    float dist = sqrt(dot(abDiff, metricAt(middle) * abDiff));

    int segmentCount = 1;
    float err = 1.0;
    while(err > 1e-4)
    {
        segmentCount *= 2;
        vec3 segBeg = a;
        vec3 ds = abDiff / float(segmentCount);
        vec3 half_ds = ds / 2.0;

        float newDist = 0.0;
        for(int i=0; i < segmentCount; ++i)
        {
            mat3 metric = metricAt(segBeg + half_ds);
            newDist += sqrt(dot(ds, metric * ds));
            segBeg += ds;
        }

        err = abs(newDist - dist);
        dist = newDist;
    }

    return dist;
}

vec3 riemannianSegment(in vec3 a, in vec3 b)
{
    return normalize(b - a) *
        riemannianDistance(a, b);
}


// Element Volume
float tetVolume(in vec3 vp[TET_VERTEX_COUNT])
{
    float detSum = determinant(mat3(
        riemannianSegment(vp[3], vp[0]),
        riemannianSegment(vp[3], vp[1]),
        riemannianSegment(vp[3], vp[2])));

    return detSum / 6.0;
}

float priVolume(in vec3 vp[PRI_VERTEX_COUNT])
{
    vec3 e20 = riemannianSegment(vp[2], vp[0]);
    vec3 e21 = riemannianSegment(vp[2], vp[1]);
    vec3 e23 = riemannianSegment(vp[2], vp[3]);
    vec3 e24 = riemannianSegment(vp[2], vp[4]);
    vec3 e25 = riemannianSegment(vp[2], vp[5]);

    float detSum = 0.0;
    detSum += determinant(mat3(e24, e20, e21));
    detSum += determinant(mat3(e25, e21, e23));
    detSum += determinant(mat3(e24, e21, e25));

    return detSum / 6.0;
}

float hexVolume(in vec3 vp[HEX_VERTEX_COUNT])
{
    float detSum = 0.0;
    detSum += determinant(mat3(
        riemannianSegment(vp[0], vp[1]),
        riemannianSegment(vp[0], vp[2]),
        riemannianSegment(vp[0], vp[4])));
    detSum += determinant(mat3(
        riemannianSegment(vp[3], vp[1]),
        riemannianSegment(vp[3], vp[7]),
        riemannianSegment(vp[3], vp[2])));
    detSum += determinant(mat3(
        riemannianSegment(vp[5], vp[1]),
        riemannianSegment(vp[5], vp[4]),
        riemannianSegment(vp[5], vp[7])));
    detSum += determinant(mat3(
        riemannianSegment(vp[6], vp[2]),
        riemannianSegment(vp[6], vp[7]),
        riemannianSegment(vp[6], vp[4])));
    detSum += determinant(mat3(
        riemannianSegment(vp[1], vp[2]),
        riemannianSegment(vp[1], vp[4]),
        riemannianSegment(vp[1], vp[7])));

    return detSum / 6.0;
}


// High level measurement
vec3 computeSpringForce(in vec3 pi, in vec3 pj)
{
    if(pi == pj)
        return vec3(0.0);

    float d = riemannianDistance(pi, pj);
    vec3 u = (pi - pj) / d;

    float d2 = d * d;
    float d4 = d2 * d2;

    //float f = (1 - d4) * exp(-d4);
    //float f = (1-d2)*exp(-d2/4.0)/2.0;
    float f = (1-d2)*exp(-abs(d)/(sqrt(2.0)));

    return f * u;
}

vec3 computeVertexEquilibrium(in uint vId)
{
    Topo topo = topos[vId];
    vec3 pos = vec3(verts[vId].p);

    vec3 forceTotal = vec3(0.0);
    uint neigVertCount = topo.neigVertCount;
    for(uint i=0, n = topo.neigVertBase; i < neigVertCount; ++i, ++n)
    {
        NeigVert neigVert = neigVerts[n];
        vec3 npos = vec3(verts[neigVert.v].p);

        forceTotal += computeSpringForce(pos, npos);
    }

    vec3 equilibrium = pos + forceTotal;
    return equilibrium;
}
