struct LocalTet
{
    uint v[4];
    uint n[4];
};


layout(shared, binding = LOCAL_TETS_BUFFER_BINDING) buffer LocalTets
{
    LocalTet localTets[];
};

layout(shared, binding = LOCAL_CACHE_BUFFER_BINDING) buffer LocalCache
{
    uint localCache[];
};

bool tetParams(in uint v[4], in vec3 p, out float coor[4]);


subroutine mat3 metricAtSub(in vec3 position, in uint vId);
layout(location=0) subroutine uniform metricAtSub metricAtUni;

mat3 metricAt(in vec3 position, in uint vId)
{
    return metricAtUni(position, vId);
}


const uint MAX_TABOO = 32;
bool isTaboo(uint tId, uint taboo[MAX_TABOO], uint count)
{
    if(tId != -1)
    {
        for(uint i=0; i < count; ++i)
            if(tId == taboo[i])
                return true;
    }

    return false;
}

layout(index=0) subroutine(metricAtSub)
mat3 metricAtImpl(in vec3 position, in uint cacheId)
{
    // Taboo search structures
    uint tabooCount = 0;
    uint taboo[MAX_TABOO];

    uint tetId = localCache[cacheId];
    LocalTet tet = localTets[tetId];

    float coor[4];
    while(!tetParams(tet.v, position, coor))
    {
        uint n = -1;
        float minCoor = 1/0.0;

        if(coor[0] < minCoor && !isTaboo(tet.n[0], taboo, tabooCount))
        {
            n = 0;
            minCoor = coor[0];
        }
        if(coor[1] < minCoor && !isTaboo(tet.n[1], taboo, tabooCount))
        {
            n = 1;
            minCoor = coor[1];
        }
        if(coor[2] < minCoor && !isTaboo(tet.n[2], taboo, tabooCount))
        {
            n = 2;
            minCoor = coor[2];
        }
        if(coor[3] < minCoor && !isTaboo(tet.n[3], taboo, tabooCount))
        {
            n = 3;
            minCoor = coor[3];
        }

        bool clipCurrentTet = false;
        if(n != -1)
        {
            uint nextTet = tet.n[n];

            if((nextTet != -1))
            {
                if(tabooCount < MAX_TABOO)
                {
                    // Add last tet to taboo list
                    taboo[tabooCount] = tetId;
                    ++tabooCount;

                    // Fetch the next local tet
                    tet = localTets[nextTet];
                    tetId = nextTet;
                }
                else
                {
                    // We went too far,
                    // stay where we are
                    clipCurrentTet = true;
                }
            }
            else
            {
                // The sampling point is on
                // the other side of the boundary
                clipCurrentTet = true;
                // This may not be an issue
            }
        }
        else
        {
            // Every surrounding tet
            // were already visited
            clipCurrentTet = true;
        }


        if(clipCurrentTet)
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
            break;
        }
    }

    // TODO wbussiere 2016-03-07 :
    //  Verify potential race conditions issues
    localCache[cacheId] = tetId;

    return coor[0] * mat3(refMetrics[tet.v[0]]) +
           coor[1] * mat3(refMetrics[tet.v[1]]) +
           coor[2] * mat3(refMetrics[tet.v[2]]) +
           coor[3] * mat3(refMetrics[tet.v[3]]);
}
