#include "Base.cuh"


///////////////////////////////
//   Function declarations   //
///////////////////////////////
__device__ riemannianDistanceFct riemannianDistanceImpl = nullptr;
__device__ riemannianSegmentFct riemannianSegmentImpl = nullptr;

__device__ tetVolumeFct tetVolumeImpl = nullptr;
__device__ priVolumeFct priVolumeImpl = nullptr;
__device__ hexVolumeFct hexVolumeImpl = nullptr;

__device__ computeVertexEquilibriumFct computeVertexEquilibrium = nullptr;


//////////////////////////////
//   Function definitions   //
//////////////////////////////
__device__ float tetVolume(const Tet& tet)
{
    const vec3 vp[] = {
        verts[tet.v[0]].p,
        verts[tet.v[1]].p,
        verts[tet.v[2]].p,
        verts[tet.v[3]].p,
    };

    return (*tetVolumeImpl)(vp, tet);
}

__device__ float priVolume(const Pri& pri)
{
    const vec3 vp[] = {
        verts[pri.v[0]].p,
        verts[pri.v[1]].p,
        verts[pri.v[2]].p,
        verts[pri.v[3]].p,
        verts[pri.v[4]].p,
        verts[pri.v[5]].p,
    };

    return (*priVolumeImpl)(vp, pri);
}

__device__ float hexVolume(const Hex& hex)
{
    const vec3 vp[] = {
        verts[hex.v[0]].p,
        verts[hex.v[1]].p,
        verts[hex.v[2]].p,
        verts[hex.v[3]].p,
        verts[hex.v[4]].p,
        verts[hex.v[5]].p,
        verts[hex.v[6]].p,
        verts[hex.v[7]].p
    };
    return (*hexVolumeImpl)(vp, hex);
}

__device__ float computeLocalElementSize(uint vId)
{
    vec3 pos = verts[vId].p;
    Topo topo = topos[vId];

    float totalSize = 0.0;
    uint neigVertCount = topo.neigVertCount;
    for(uint i=0, n = topo.neigVertBase; i < neigVertCount; ++i, ++n)
    {
        totalSize += length(pos - verts[neigVerts[n].v].p);
    }

    return totalSize / neigVertCount;
}
