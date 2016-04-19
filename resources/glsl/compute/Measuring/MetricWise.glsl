mat3 metricAt(in vec3 position, inout uint cachedRefTet);

// Distances
float riemannianDistance(in vec3 a, in vec3 b, inout uint cachedRefTet)
{
    vec3 abDiff = b - a;
    vec3 middle = (a + b) / 2.0;
    float dist = sqrt(dot(abDiff, metricAt(middle, cachedRefTet) * abDiff));

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
            mat3 metric = metricAt(segBeg + half_ds, cachedRefTet);
            newDist += sqrt(dot(ds, metric * ds));
            segBeg += ds;
        }

        err = abs(newDist - dist);
        dist = newDist;
    }

    return dist;
}

vec3 riemannianSegment(in vec3 a, in vec3 b, inout uint cachedRefTet)
{
    return normalize(b - a) *
        riemannianDistance(a, b, cachedRefTet);
}


// Element Volume
float tetVolume(in vec3 vp[TET_VERTEX_COUNT], inout Tet tet)
{
    float detSum = determinant(mat3(
        riemannianSegment(vp[0], vp[3], tet.c[0]),
        riemannianSegment(vp[1], vp[3], tet.c[0]),
        riemannianSegment(vp[2], vp[3], tet.c[0])));

    return detSum / 6.0;
}

float priVolume(in vec3 vp[PRI_VERTEX_COUNT], inout Pri pri)
{
    vec3 e02 = riemannianSegment(vp[0], vp[2], pri.c[0]);
    vec3 e12 = riemannianSegment(vp[1], vp[2], pri.c[1]);
    vec3 e32 = riemannianSegment(vp[3], vp[2], pri.c[3]);
    vec3 e42 = riemannianSegment(vp[4], vp[2], pri.c[4]);
    vec3 e52 = riemannianSegment(vp[5], vp[2], pri.c[5]);

    float detSum = 0.0;
    detSum += determinant(mat3(e32, e52, e42));
    detSum += determinant(mat3(e02, e32, e42));
    detSum += determinant(mat3(e12, e02, e42));

    return detSum / 6.0;
}

float hexVolume(in vec3 vp[HEX_VERTEX_COUNT], inout Hex hex)
{
    float detSum = 0.0;
    detSum += determinant(mat3(
        riemannianSegment(vp[1], vp[0], hex.c[1]),
        riemannianSegment(vp[4], vp[0], hex.c[4]),
        riemannianSegment(vp[3], vp[0], hex.c[3])));
    detSum += determinant(mat3(
        riemannianSegment(vp[1], vp[2], hex.c[1]),
        riemannianSegment(vp[3], vp[2], hex.c[3]),
        riemannianSegment(vp[6], vp[2], hex.c[6])));
    detSum += determinant(mat3(
        riemannianSegment(vp[4], vp[5], hex.c[4]),
        riemannianSegment(vp[1], vp[5], hex.c[1]),
        riemannianSegment(vp[6], vp[5], hex.c[6])));
    detSum += determinant(mat3(
        riemannianSegment(vp[4], vp[7], hex.c[4]),
        riemannianSegment(vp[6], vp[7], hex.c[6]),
        riemannianSegment(vp[3], vp[7], hex.c[3])));
    detSum += determinant(mat3(
        riemannianSegment(vp[1], vp[4], hex.c[1]),
        riemannianSegment(vp[6], vp[4], hex.c[6]),
        riemannianSegment(vp[3], vp[4], hex.c[3])));

    return detSum / 6.0;
}


// High level measurement
vec3 computeSpringForce(in vec3 pi, in vec3 pj, inout uint cachedRefTet)
{
    if(pi == pj)
        return vec3(0.0);

    float d = riemannianDistance(pi, pj, cachedRefTet);
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
    vec3 pos = verts[vId].p;

    vec3 forceTotal = vec3(0.0);
    uint neigVertCount = topo.neigVertCount;
    for(uint i=0, n = topo.neigVertBase; i < neigVertCount; ++i, ++n)
    {
        NeigVert neigVert = neigVerts[n];
        vec3 npos = verts[neigVert.v].p;

        forceTotal += computeSpringForce(pos, npos, verts[vId].c);
    }

    vec3 equilibrium = pos + forceTotal;
    return equilibrium;
}
