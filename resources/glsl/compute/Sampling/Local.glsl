struct LocalTet
{
    uint v[4];
    uint n[4];
};


layout(shared, binding = LOCAL_TETS_BUFFER_BINDING) buffer LocalTets
{
    LocalTet localTets[];
};

bool tetParams(in uint v[4], in vec3 p, out float coor[4]);

bool triIntersect(
        in vec3 v1, in vec3 v2, in vec3 v3,
        in vec3 orig, in vec3 dir);


subroutine mat3 metricAtSub(in vec3 position, inout uint cachedRefTet);
layout(location=METRIC_AT_SUBROUTINE_LOC)
subroutine uniform metricAtSub metricAtUni;

mat3 metricAt(in vec3 position, inout uint cachedRefTet)
{
    return metricAtUni(position, cachedRefTet);
}


layout(index=METRIC_AT_SUBROUTINE_IDX) subroutine(metricAtSub)
mat3 metricAtImpl(in vec3 position, inout uint cachedRefTet)
{
    LocalTet tet = localTets[cachedRefTet];

    float coor[4];
    int visitedTet = 0;
    bool outOfTet = false;
    bool outOfBounds = false;

    while(!outOfBounds && !tetParams(tet.v, position, coor))
    {
        vec3 orig, dir;

        if(visitedTet == 0)
        {
            orig = 0.25f * (
                refVerts[tet.v[0]].p +
                refVerts[tet.v[1]].p +
                refVerts[tet.v[2]].p +
                refVerts[tet.v[3]].p);

            dir = normalize(orig - position);
        }


        int t=0;
        for(;t < 4; ++t)
        {
            if(triIntersect(
                refVerts[tet.v[MeshTet_tris[t].v[0]]].p,
                refVerts[tet.v[MeshTet_tris[t].v[1]]].p,
                refVerts[tet.v[MeshTet_tris[t].v[2]]].p,
                orig, dir))
            {
                if(tet.n[t] != -1)
                {
                    ++visitedTet;
                    tet = localTets[tet.n[t]];
                }
                else
                {
                    outOfBounds = true;
                    outOfTet = true;
                }

                break;
            }
        }

        if(t == 4)
        {
            if(visitedTet == 0)
            {
                outOfTet = true;
                break;
            }
            else
            {
                uint n = tet.n[0];
                float c = coor[0];

                if(coor[1] < c) {n = tet.n[1]; c = coor[1];}
                if(coor[2] < c) {n = tet.n[2]; c = coor[2];}
                if(coor[3] < c) {n = tet.n[3]; c = coor[3];}

                if(n != -1)
                {
                    visitedTet = 0;
                    tet = localTets[n];
                }
                else
                {
                    outOfTet = true;
                    outOfBounds = true;
                }
            }
        }
    }

    if(outOfTet)
    {
        // Clamp sample to current tet
        // It's seems to be the closest
        // we can get to the sampling point
        float sum = 0.0;
        if(coor[0] < 0.0) coor[0] = 0.0; else sum += coor[0];
        if(coor[1] < 0.0) coor[1] = 0.0; else sum += coor[1];
        if(coor[2] < 0.0) coor[2] = 0.0; else sum += coor[2];
        if(coor[3] < 0.0) coor[3] = 0.0; else sum += coor[3];
        coor[0] /= sum;
        coor[1] /= sum;
        coor[2] /= sum;
        coor[3] /= sum;
    }

    return coor[0] * mat3(refMetrics[tet.v[0]]) +
           coor[1] * mat3(refMetrics[tet.v[1]]) +
           coor[2] * mat3(refMetrics[tet.v[2]]) +
           coor[3] * mat3(refMetrics[tet.v[3]]);
}
