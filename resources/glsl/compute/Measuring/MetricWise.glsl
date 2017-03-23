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
float tetVolume(in vec3 vp[PARAM_VERTEX_COUNT], inout Tet tet)
{
    float detSum = determinant(mat3(
        riemannianSegment(vp[0], vp[3], tet.c[0]),
        riemannianSegment(vp[1], vp[3], tet.c[0]),
        riemannianSegment(vp[2], vp[3], tet.c[0])));

    return detSum / 6.0;
}

float priVolume(in vec3 vp[PARAM_VERTEX_COUNT], inout Pri pri)
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

float hexVolume(in vec3 vp[PARAM_VERTEX_COUNT], inout Hex hex)
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
void sumNode(
        inout float totalWeight,
        inout vec3 displacement,
        in vec3 pos,
        inout Vert v)
{
    vec3 d = v.p - pos;

    if(d != vec3(0))
    {
        vec3 n = normalize(d);
        mat3 M = metricAt(v.p, v.c);
        float weight = sqrt(dot(n, M * n));

        totalWeight += weight;
        displacement += weight * d;
    }
}

vec3 computeVertexEquilibrium(in uint vId)
{
    Topo topo = topos[vId];

    float totalWeight = 0.0f;
    vec3 displacement = vec3(0.0);
    vec3 pos = verts[vId].p;

    uint neigElemCount = topo.neigElemCount;
    for(uint i=0, n = topo.neigElemBase; i<neigElemCount; ++i, ++n)
    {
        NeigElem neigElem = neigElems[n];

        switch(neigElem.type)
        {
        case TET_ELEMENT_TYPE:
            for(uint i=0; i < TET_VERTEX_COUNT; ++i)
                sumNode(totalWeight, displacement, pos,
                        verts[tets[neigElem.id].v[i]]);
            break;

        case PRI_ELEMENT_TYPE:
            for(uint i=0; i < PRI_VERTEX_COUNT; ++i)
                sumNode(totalWeight, displacement, pos,
                        verts[pris[neigElem.id].v[i]]);
            break;

        case HEX_ELEMENT_TYPE:
            for(uint i=0; i < HEX_VERTEX_COUNT; ++i)
                sumNode(totalWeight, displacement, pos,
                        verts[hexs[neigElem.id].v[i]]);
            break;
        }
    }

    return pos + displacement / totalWeight;
}
