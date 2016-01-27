#include "Base.cuh"


///////////////////////////////
//   Function declarations   //
///////////////////////////////
__device__ riemannianDistanceFct riemannianDistanceImpl = nullptr;
__device__ riemannianSegmentFct riemannianSegmentImpl = nullptr;

__device__ tetVolumeFct tetVolumeImpl = nullptr;
__device__ priVolumeFct priVolumeImpl = nullptr;
__device__ hexVolumeFct hexVolumeImpl = nullptr;

__device__ computeVertexEquilibriumFct computeVertexEquilibriumImpl = nullptr;


//////////////////////////////
//   Function definitions   //
//////////////////////////////
__device__ float tetVolume(const Tet& tet)
{
    const vec3 vp[] = {
        vec3(verts[tet.v[0]].p),
        vec3(verts[tet.v[1]].p),
        vec3(verts[tet.v[2]].p),
        vec3(verts[tet.v[3]].p),
    };

    return (*tetVolumeImpl)(vp);
}

__device__ float priVolume(const Pri& pri)
{
    const vec3 vp[] = {
        vec3(verts[pri.v[0]].p),
        vec3(verts[pri.v[1]].p),
        vec3(verts[pri.v[2]].p),
        vec3(verts[pri.v[3]].p),
        vec3(verts[pri.v[4]].p),
        vec3(verts[pri.v[5]].p),
    };

    return (*priVolumeImpl)(vp);
}

__device__ float hexVolume(const Hex& hex)
{
    const vec3 vp[] = {
        vec3(verts[hex.v[0]].p),
        vec3(verts[hex.v[1]].p),
        vec3(verts[hex.v[2]].p),
        vec3(verts[hex.v[3]].p),
        vec3(verts[hex.v[4]].p),
        vec3(verts[hex.v[5]].p),
        vec3(verts[hex.v[6]].p),
        vec3(verts[hex.v[7]].p)
    };
    return (*hexVolumeImpl)(vp);
}

__device__ float computeLocalElementSize(uint vId)
{
    vec3 pos = vec3(verts[vId].p);
    Topo topo = topos[vId];

    float totalSize = 0.0;
    uint neigVertCount = topo.neigVertCount;
    for(uint i=0, n = topo.neigVertBase; i < neigVertCount; ++i, ++n)
    {
        totalSize += length(pos - vec3(verts[neigVerts[n].v].p));
    }

    return totalSize / neigVertCount;
}
