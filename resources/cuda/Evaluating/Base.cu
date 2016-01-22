#include "../Mesh.cuh"


///////////////////////////////
//   Function declarations   //
///////////////////////////////

// Externally defined
__device__ float tetQuality(const vec3 vp[TET_VERTEX_COUNT]);
__device__ float priQuality(const vec3 vp[PRI_VERTEX_COUNT]);
__device__ float hexQuality(const vec3 vp[HEX_VERTEX_COUNT]);

// Internally defined
__device__ float tetQuality(const Tet &tet);
__device__ float priQuality(const Pri &pri);
__device__ float hexQuality(const Hex &hex);

__device__ float patchQuality(uint vId);


//////////////////////////////
//   Function definitions   //
//////////////////////////////

// Element Quality
__device__ float tetQuality(const Tet& tet)
{
    const vec3 vp[] = {
        vec3(verts[tet.v[0]].p),
        vec3(verts[tet.v[1]].p),
        vec3(verts[tet.v[2]].p),
        vec3(verts[tet.v[3]].p)
    };

    return tetQuality(vp);
}

__device__ float priQuality(const Pri& pri)
{
    const vec3 vp[] = {
        vec3(verts[pri.v[0]].p),
        vec3(verts[pri.v[1]].p),
        vec3(verts[pri.v[2]].p),
        vec3(verts[pri.v[3]].p),
        vec3(verts[pri.v[4]].p),
        vec3(verts[pri.v[5]].p)
    };

    return priQuality(vp);
}

__device__ float hexQuality(const Hex& hex)
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

    return hexQuality(vp);
}

__device__ void accumulatePatchQuality(
        float& patchQuality,
        float& patchWeight,
        float elemQuality)
{
    patchQuality = min(
        min(patchQuality, elemQuality),  // If sign(patch) != sign(elem)
        min(patchQuality * elemQuality,  // If sign(patch) & sign(elem) > 0
            patchQuality + elemQuality));// If sign(patch) & sign(elem) < 0
}

__device__ float finalizePatchQuality(float patchQuality, float patchWeight)
{
    return patchQuality;
}

__device__ float patchQuality(uint vId)
{
    Topo topo = topos[vId];

    float patchWeight = 0.0;
    float patchQuality = 1.0;
    uint neigElemCount = topo.neigElemCount;
    for(uint i=0, n = topo.neigElemBase; i < neigElemCount; ++i, ++n)
    {
        NeigElem neigElem = neigElems[n];

        switch(neigElem.type)
        {
        case TET_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                tetQuality(tets[neigElem.id]));
            break;

        case PRI_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                priQuality(pris[neigElem.id]));
            break;

        case HEX_ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                hexQuality(hexs[neigElem.id]));
            break;
        }
    }

    return finalizePatchQuality(patchQuality, patchWeight);
}
