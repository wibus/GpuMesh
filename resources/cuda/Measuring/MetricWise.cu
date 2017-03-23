#include "Base.cuh"


// Distance
__device__ float metricWiseRiemannianDistance(const vec3& a, const vec3& b, uint& cachedRefTet)
{
    vec3 abDiff = b - a;
    vec3 middle = (a + b) / 2.0f;
    float dist = sqrt(dot(abDiff, metricAt(middle, cachedRefTet) * abDiff));

    int segmentCount = 1;
    float err = 1.0;
    while(err > 1e-4)
    {
        segmentCount *= 2;
        vec3 segBeg = a;
        vec3 ds = abDiff / float(segmentCount);
        vec3 half_ds = ds / 2.0f;

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

__device__ vec3 metricWiseRiemannianSegment(const vec3& a, const vec3& b, uint& cachedRefTet)
{
    return normalize(b - a) *
        metricWiseRiemannianDistance(a, b, cachedRefTet);
}


// Volume
__device__ float metricWiseTetVolume(const vec3 vp[TET_VERTEX_COUNT], const Tet& tet)
{
    float detSum = determinant(mat3(
        (*riemannianSegmentImpl)(vp[0], vp[3], tet.c[0]),
        (*riemannianSegmentImpl)(vp[1], vp[3], tet.c[0]),
        (*riemannianSegmentImpl)(vp[2], vp[3], tet.c[0])));

    return detSum / 6.0;
}

__device__ float metricWisePriVolume(const vec3 vp[PRI_VERTEX_COUNT], const Pri& pri)
{
    vec3 e02 = (*riemannianSegmentImpl)(vp[0], vp[2], pri.c[0]);
    vec3 e12 = (*riemannianSegmentImpl)(vp[1], vp[2], pri.c[1]);
    vec3 e32 = (*riemannianSegmentImpl)(vp[3], vp[2], pri.c[3]);
    vec3 e42 = (*riemannianSegmentImpl)(vp[4], vp[2], pri.c[4]);
    vec3 e52 = (*riemannianSegmentImpl)(vp[5], vp[2], pri.c[5]);

    float detSum = 0.0;
    detSum += determinant(mat3(e32, e52, e42));
    detSum += determinant(mat3(e02, e32, e42));
    detSum += determinant(mat3(e12, e02, e42));

    return detSum / 6.0;
}

__device__ float metricWiseHexVolume(const vec3 vp[HEX_VERTEX_COUNT], const Hex& hex)
{
    float detSum = 0.0;
    detSum += determinant(mat3(
        (*riemannianSegmentImpl)(vp[1], vp[0], hex.c[1]),
        (*riemannianSegmentImpl)(vp[4], vp[0], hex.c[4]),
        (*riemannianSegmentImpl)(vp[3], vp[0], hex.c[3])));
    detSum += determinant(mat3(
        (*riemannianSegmentImpl)(vp[1], vp[2], hex.c[1]),
        (*riemannianSegmentImpl)(vp[3], vp[2], hex.c[3]),
        (*riemannianSegmentImpl)(vp[6], vp[2], hex.c[6])));
    detSum += determinant(mat3(
        (*riemannianSegmentImpl)(vp[4], vp[5], hex.c[4]),
        (*riemannianSegmentImpl)(vp[1], vp[5], hex.c[1]),
        (*riemannianSegmentImpl)(vp[6], vp[5], hex.c[6])));
    detSum += determinant(mat3(
        (*riemannianSegmentImpl)(vp[4], vp[7], hex.c[4]),
        (*riemannianSegmentImpl)(vp[6], vp[7], hex.c[6]),
        (*riemannianSegmentImpl)(vp[3], vp[7], hex.c[3])));
    detSum += determinant(mat3(
        (*riemannianSegmentImpl)(vp[1], vp[4], hex.c[1]),
        (*riemannianSegmentImpl)(vp[6], vp[4], hex.c[6]),
        (*riemannianSegmentImpl)(vp[3], vp[4], hex.c[3])));

    return detSum / 6.0;
}


// High level measurement
__device__ void sumNode(
        float& totalWeight,
        vec3& displacement,
        const vec3& pos,
        const Vert& v)
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

__device__ vec3 metricWiseComputeVertexEquilibrium(uint vId)
{
    const Topo& topo = topos[vId];

    float totalWeight = 0.0f;
    vec3 displacement = vec3(0.0);
    const vec3& pos = verts[vId].p;

    uint neigElemCount = topo.neigElemCount;
    for(uint i=0, n = topo.neigElemBase; i<neigElemCount; ++i, ++n)
    {
        const NeigElem& neigElem = neigElems[n];

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


// Fonction pointers
__device__ riemannianDistanceFct metricWiseRiemannianDistancePtr = metricWiseRiemannianDistance;
__device__ riemannianSegmentFct metricWiseRiemannianSegmentPtr = metricWiseRiemannianSegment;

__device__ tetVolumeFct metricWiseTetVolumePtr = metricWiseTetVolume;
__device__ priVolumeFct metricWisePriVolumePtr = metricWisePriVolume;
__device__ hexVolumeFct metricWiseHexVolumePtr = metricWiseHexVolume;

__device__ computeVertexEquilibriumFct metricWiseComputeVertexEquilibriumPtr = metricWiseComputeVertexEquilibrium;


// CUDA Drivers
void installCudaMetricWiseMeasurer()
{
    riemannianDistanceFct d_riemannianDistance = nullptr;
    cudaMemcpyFromSymbol(&d_riemannianDistance, metricWiseRiemannianDistancePtr, sizeof(riemannianDistanceFct));
    cudaMemcpyToSymbol(riemannianDistanceImpl, &d_riemannianDistance, sizeof(riemannianDistanceFct));

    riemannianSegmentFct d_riemannianSegment = nullptr;
    cudaMemcpyFromSymbol(&d_riemannianSegment, metricWiseRiemannianSegmentPtr, sizeof(riemannianSegmentFct));
    cudaMemcpyToSymbol(riemannianSegmentImpl, &d_riemannianSegment, sizeof(riemannianSegmentFct));


    tetVolumeFct d_tetVolume = nullptr;
    cudaMemcpyFromSymbol(&d_tetVolume, metricWiseTetVolumePtr, sizeof(tetVolumeFct));
    cudaMemcpyToSymbol(tetVolumeImpl, &d_tetVolume, sizeof(tetVolumeFct));

    priVolumeFct d_priVolume = nullptr;
    cudaMemcpyFromSymbol(&d_priVolume, metricWisePriVolumePtr, sizeof(priVolumeFct));
    cudaMemcpyToSymbol(priVolumeImpl, &d_priVolume, sizeof(priVolumeFct));

    hexVolumeFct d_hexVolume = nullptr;
    cudaMemcpyFromSymbol(&d_hexVolume, metricWiseHexVolumePtr, sizeof(hexVolumeFct));
    cudaMemcpyToSymbol(hexVolumeImpl, &d_hexVolume, sizeof(hexVolumeFct));


    computeVertexEquilibriumFct d_computeVertexEquilibrium = nullptr;
    cudaMemcpyFromSymbol(&d_computeVertexEquilibrium, metricWiseComputeVertexEquilibriumPtr, sizeof(computeVertexEquilibriumFct));
    cudaMemcpyToSymbol(computeVertexEquilibrium, &d_computeVertexEquilibrium, sizeof(computeVertexEquilibriumFct));


    if(verboseCuda)
        printf("I -> CUDA \tMetric Wise Measurer installed\n");
}
