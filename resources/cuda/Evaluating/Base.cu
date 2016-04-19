#include "Base.cuh"


///////////////////////////////
//   Function declarations   //
///////////////////////////////
__device__ tetQualityFct tetQualityImpl;
__device__ priQualityFct priQualityImpl;
__device__ hexQualityFct hexQualityImpl;


//////////////////////////////
//   Function definitions   //
//////////////////////////////
__device__ float tetQuality(const Tet& tet)
{
    const vec3 vp[] = {
        verts[tet.v[0]].p,
        verts[tet.v[1]].p,
        verts[tet.v[2]].p,
        verts[tet.v[3]].p
    };

    return (*tetQualityImpl)(vp, tet);
}

__device__ float priQuality(const Pri& pri)
{
    const vec3 vp[] = {
        verts[pri.v[0]].p,
        verts[pri.v[1]].p,
        verts[pri.v[2]].p,
        verts[pri.v[3]].p,
        verts[pri.v[4]].p,
        verts[pri.v[5]].p
    };

    return (*priQualityImpl)(vp, pri);
}

__device__ float hexQuality(const Hex& hex)
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

    return (*hexQualityImpl)(vp, hex);
}

__device__ void accumulatePatchQuality(
        double& patchQuality,
        double& patchWeight,
        double elemQuality)
{
    patchQuality = min(patchQuality, elemQuality);
    /*
    patchQuality = min(
        min(patchQuality, elemQuality),  // If sign(patch) != sign(elem)
        min(patchQuality * elemQuality,  // If sign(patch) & sign(elem) > 0
            patchQuality + elemQuality));// If sign(patch) & sign(elem) < 0
            */
}

__device__ float finalizePatchQuality(double patchQuality, double patchWeight)
{
    return float(patchQuality);
    /*
    double s = sign(patchQuality);
    return float(s * sqrt(s*patchQuality));
    */
}

__device__ float patchQuality(uint vId)
{
    Topo topo = topos[vId];

    double patchWeight = 0.0;
    double patchQuality = 1.0;
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
