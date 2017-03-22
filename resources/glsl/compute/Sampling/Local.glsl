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
    bool isUnreachable = false;

    while(!isUnreachable && !tetParams(tet.v, position, coor))
    {
        vec3 vp[] = {
            refVerts[tet.v[0]].p,
            refVerts[tet.v[1]].p,
            refVerts[tet.v[2]].p,
            refVerts[tet.v[3]].p
        };

        vec3 orig = 0.25 * (vp[0] + vp[1] + vp[2] + vp[3]);
        vec3 dir = normalize(orig - position);

        int t = 4;
        int trialCount = -1;
        while(t == 4 && !isUnreachable)
        {
            // Find exit face
            for(t = 0; t < 4; ++t)
            {
                if(triIntersect(
                    vp[MeshTet_tris[t].v[0]],
                    vp[MeshTet_tris[t].v[1]],
                    vp[MeshTet_tris[t].v[2]],
                    orig, dir))
                {
                    if(tet.n[t] != -1)
                        tet = localTets[tet.n[t]];
                    else
                        isUnreachable = true;

                    break;
                }
            }

            // If exit face not found
            if(t == 4)
            {
                // Start from an other position in the tet
                ++trialCount;

                // If there are still untried positions
                if(trialCount < 4)
                {
                    const float INV_MASS = 1.0 / 10.0;
                    const float WEIGHTS[] = {1.0, 2.0, 3.0, 4.0};

                    // Initialize ray from next position
                    orig = INV_MASS * (
                        WEIGHTS[(trialCount + 0) % 4] * vp[0] +
                        WEIGHTS[(trialCount + 1) % 4] * vp[1] +
                        WEIGHTS[(trialCount + 2) % 4] * vp[2] +
                        WEIGHTS[(trialCount + 3) % 4] * vp[3]);

                    dir = normalize(orig - position);
                }
                else
                {
                    isUnreachable = true;
                    break;
                }
            }
        }
    }


    if(isUnreachable)
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
