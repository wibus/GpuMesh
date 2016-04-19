#include "Base.cuh"


__device__ float Lambda;


// Vertex Accum
__device__ void addPosition(uint vId, vec3 pos, float weight);


// ENTRY POINTS //
__device__ void getmeSmoothTet(uint eId)
{
    const Tet& tet = tets[eId];

    uint vi[] = {
        tet.v[0],
        tet.v[1],
        tet.v[2],
        tet.v[3]
    };

    vec3 vp[] = {
        verts[vi[0]].p,
        verts[vi[1]].p,
        verts[vi[2]].p,
        verts[vi[3]].p
    };

    vec3 n[] = {
        cross(vp[3]-vp[1], vp[2]-vp[1]),
        cross(vp[3]-vp[2], vp[0]-vp[2]),
        cross(vp[1]-vp[3], vp[0]-vp[3]),
        cross(vp[1]-vp[0], vp[2]-vp[0]),
    };

    vec3 vpp[] = {
        vp[0] + n[0] * (Lambda / sqrt(length(n[0]))),
        vp[1] + n[1] * (Lambda / sqrt(length(n[1]))),
        vp[2] + n[2] * (Lambda / sqrt(length(n[2]))),
        vp[3] + n[3] * (Lambda / sqrt(length(n[3])))
    };


    float volume = tetVolumeImpl(vp, tet);
    float volumePrime = tetVolumeImpl(vpp, tet);
    float absVolumeRation = abs(volume / volumePrime);
    float volumeVar = pow(absVolumeRation, 1.0/3.0);

    vec3 center = float(1.0/4.0) * (
        vp[0] + vp[1] + vp[2] + vp[3]);

    vpp[0] = center + volumeVar * (vpp[0] - center);
    vpp[1] = center + volumeVar * (vpp[1] - center);
    vpp[2] = center + volumeVar * (vpp[2] - center);
    vpp[3] = center + volumeVar * (vpp[3] - center);

    if(topos[vi[0]].type > 0) vpp[0] = snapToBoundary(topos[vi[0]].type, vpp[0]);
    if(topos[vi[1]].type > 0) vpp[1] = snapToBoundary(topos[vi[1]].type, vpp[1]);
    if(topos[vi[2]].type > 0) vpp[2] = snapToBoundary(topos[vi[2]].type, vpp[2]);
    if(topos[vi[3]].type > 0) vpp[3] = snapToBoundary(topos[vi[3]].type, vpp[3]);

    float quality = tetQualityImpl(vp, tet);
    float qualityPrime = tetQualityImpl(vpp, tet);

    float weight = qualityPrime / (1.0 + quality);
    addPosition(vi[0], vpp[0], weight);
    addPosition(vi[1], vpp[1], weight);
    addPosition(vi[2], vpp[2], weight);
    addPosition(vi[3], vpp[3], weight);
}

__device__ void getmeSmoothPri(uint eId)
{
    const Pri& pri = pris[eId];

    uint vi[] = {
        pri.v[0],
        pri.v[1],
        pri.v[2],
        pri.v[3],
        pri.v[4],
        pri.v[5]
    };

    vec3 vp[] = {
        verts[vi[0]].p,
        verts[vi[1]].p,
        verts[vi[2]].p,
        verts[vi[3]].p,
        verts[vi[4]].p,
        verts[vi[5]].p
    };

    vec3 aux[] = {
        (vp[0] + vp[1] + vp[2]) / 3.0f,
        (vp[0] + vp[1] + vp[4] + vp[3]) / 4.0f,
        (vp[1] + vp[2] + vp[5] + vp[4]) / 4.0f,
        (vp[2] + vp[0] + vp[3] + vp[5]) / 4.0f,
        (vp[3] + vp[4] + vp[5]) / 3.0f,
    };

    vec3 n[] = {
        cross(aux[1] - aux[0], aux[3] - aux[0]),
        cross(aux[2] - aux[0], aux[1] - aux[0]),
        cross(aux[3] - aux[0], aux[2] - aux[0]),
        cross(aux[3] - aux[4], aux[1] - aux[4]),
        cross(aux[1] - aux[4], aux[2] - aux[4]),
        cross(aux[2] - aux[4], aux[3] - aux[4]),
    };

    float t = (4.0/5.0) * (1.0 - pow(4.0/39.0, 0.25) * Lambda);
    float it = 1.0 - t;
    vec3 bases[] = {
        it * aux[0] + t * (aux[3] + aux[1]) / 2.0f,
        it * aux[0] + t * (aux[1] + aux[2]) / 2.0f,
        it * aux[0] + t * (aux[2] + aux[3]) / 2.0f,
        it * aux[4] + t * (aux[3] + aux[1]) / 2.0f,
        it * aux[4] + t * (aux[1] + aux[2]) / 2.0f,
        it * aux[4] + t * (aux[2] + aux[3]) / 2.0f,
    };


    // New positions
    vec3 vpp[] = {
        bases[0] + n[0] * (Lambda / sqrt(length(n[0]))),
        bases[1] + n[1] * (Lambda / sqrt(length(n[1]))),
        bases[2] + n[2] * (Lambda / sqrt(length(n[2]))),
        bases[3] + n[3] * (Lambda / sqrt(length(n[3]))),
        bases[4] + n[4] * (Lambda / sqrt(length(n[4]))),
        bases[5] + n[5] * (Lambda / sqrt(length(n[5])))
    };


    float volume = priVolumeImpl(vp, pri);
    float volumePrime = priVolumeImpl(vpp, pri);
    float absVolumeRation = abs(volume / volumePrime);
    float volumeVar = pow(absVolumeRation, 1.0/3.0);

    vec3 center = float(1.0/6.0) * (
        vp[0] + vp[1] + vp[2] + vp[3] + vp[4] + vp[5]);

    vpp[0] = center + volumeVar * (vpp[0] - center);
    vpp[1] = center + volumeVar * (vpp[1] - center);
    vpp[2] = center + volumeVar * (vpp[2] - center);
    vpp[3] = center + volumeVar * (vpp[3] - center);
    vpp[4] = center + volumeVar * (vpp[4] - center);
    vpp[5] = center + volumeVar * (vpp[5] - center);

    if(topos[vi[0]].type > 0) vpp[0] = snapToBoundary(topos[vi[0]].type, vpp[0]);
    if(topos[vi[1]].type > 0) vpp[1] = snapToBoundary(topos[vi[1]].type, vpp[1]);
    if(topos[vi[2]].type > 0) vpp[2] = snapToBoundary(topos[vi[2]].type, vpp[2]);
    if(topos[vi[3]].type > 0) vpp[3] = snapToBoundary(topos[vi[3]].type, vpp[3]);
    if(topos[vi[4]].type > 0) vpp[4] = snapToBoundary(topos[vi[4]].type, vpp[4]);
    if(topos[vi[5]].type > 0) vpp[5] = snapToBoundary(topos[vi[5]].type, vpp[5]);


    float quality = priQualityImpl(vp, pri);
    float qualityPrime = priQualityImpl(vpp, pri);

    float weight = qualityPrime / (1.0 + quality);
    addPosition(vi[0], vpp[0], weight);
    addPosition(vi[1], vpp[1], weight);
    addPosition(vi[2], vpp[2], weight);
    addPosition(vi[3], vpp[3], weight);
    addPosition(vi[4], vpp[4], weight);
    addPosition(vi[5], vpp[5], weight);
}

__device__ void getmeSmoothHex(uint eId)
{
    const Hex& hex = hexs[eId];

    uint vi[] = {
        hex.v[0],
        hex.v[1],
        hex.v[2],
        hex.v[3],
        hex.v[4],
        hex.v[5],
        hex.v[6],
        hex.v[7]
    };

    vec3 vp[] = {
        verts[vi[0]].p,
        verts[vi[1]].p,
        verts[vi[2]].p,
        verts[vi[3]].p,
        verts[vi[4]].p,
        verts[vi[5]].p,
        verts[vi[6]].p,
        verts[vi[7]].p
    };

    vec3 aux[] = {
        (vp[0] + vp[1] + vp[2] + vp[3]) / 4.0f,
        (vp[0] + vp[4] + vp[5] + vp[1]) / 4.0f,
        (vp[1] + vp[5] + vp[6] + vp[2]) / 4.0f,
        (vp[2] + vp[6] + vp[7] + vp[3]) / 4.0f,
        (vp[0] + vp[3] + vp[7] + vp[4]) / 4.0f,
        (vp[4] + vp[7] + vp[6] + vp[5]) / 4.0f,
    };

    vec3 n[] = {
        cross(aux[1] - aux[0], aux[4] - aux[0]),
        cross(aux[2] - aux[0], aux[1] - aux[0]),
        cross(aux[3] - aux[0], aux[2] - aux[0]),
        cross(aux[4] - aux[0], aux[3] - aux[0]),
        cross(aux[4] - aux[5], aux[1] - aux[5]),
        cross(aux[1] - aux[5], aux[2] - aux[5]),
        cross(aux[2] - aux[5], aux[3] - aux[5]),
        cross(aux[3] - aux[5], aux[4] - aux[5]),
    };

    vec3 bases[] = {
        (aux[0] + aux[1] + aux[4]) / 3.0f,
        (aux[0] + aux[2] + aux[1]) / 3.0f,
        (aux[0] + aux[3] + aux[2]) / 3.0f,
        (aux[0] + aux[4] + aux[3]) / 3.0f,
        (aux[5] + aux[4] + aux[1]) / 3.0f,
        (aux[5] + aux[1] + aux[2]) / 3.0f,
        (aux[5] + aux[2] + aux[3]) / 3.0f,
        (aux[5] + aux[3] + aux[4]) / 3.0f,
    };


    // New positions
    vec3 vpp[] = {
        bases[0] + n[0] * (Lambda / sqrt(length(n[0]))),
        bases[1] + n[1] * (Lambda / sqrt(length(n[1]))),
        bases[2] + n[2] * (Lambda / sqrt(length(n[2]))),
        bases[3] + n[3] * (Lambda / sqrt(length(n[3]))),
        bases[4] + n[4] * (Lambda / sqrt(length(n[4]))),
        bases[5] + n[5] * (Lambda / sqrt(length(n[5]))),
        bases[6] + n[6] * (Lambda / sqrt(length(n[6]))),
        bases[7] + n[7] * (Lambda / sqrt(length(n[7])))
    };


    float volume = hexVolumeImpl(vp, hex);
    float volumePrime = hexVolumeImpl(vpp, hex);
    float absVolumeRation = abs(volume / volumePrime);
    float volumeVar = pow(absVolumeRation, 1.0/3.0);

    vec3 center = float(1.0/8.0) * (
        vp[0] + vp[1] + vp[2] + vp[3] + vp[4] + vp[5] + vp[6] + vp[7]);

    vpp[0] = center + volumeVar * (vpp[0] - center);
    vpp[1] = center + volumeVar * (vpp[1] - center);
    vpp[2] = center + volumeVar * (vpp[2] - center);
    vpp[3] = center + volumeVar * (vpp[3] - center);
    vpp[4] = center + volumeVar * (vpp[4] - center);
    vpp[5] = center + volumeVar * (vpp[5] - center);
    vpp[6] = center + volumeVar * (vpp[6] - center);
    vpp[7] = center + volumeVar * (vpp[7] - center);

    if(topos[vi[0]].type > 0) vpp[0] = snapToBoundary(topos[vi[0]].type, vpp[0]);
    if(topos[vi[1]].type > 0) vpp[1] = snapToBoundary(topos[vi[1]].type, vpp[1]);
    if(topos[vi[2]].type > 0) vpp[2] = snapToBoundary(topos[vi[2]].type, vpp[2]);
    if(topos[vi[3]].type > 0) vpp[3] = snapToBoundary(topos[vi[3]].type, vpp[3]);
    if(topos[vi[4]].type > 0) vpp[4] = snapToBoundary(topos[vi[4]].type, vpp[4]);
    if(topos[vi[5]].type > 0) vpp[5] = snapToBoundary(topos[vi[5]].type, vpp[5]);
    if(topos[vi[6]].type > 0) vpp[6] = snapToBoundary(topos[vi[6]].type, vpp[6]);
    if(topos[vi[7]].type > 0) vpp[7] = snapToBoundary(topos[vi[7]].type, vpp[7]);


    float quality = hexQualityImpl(vp, hex);
    float qualityPrime = hexQualityImpl(vpp, hex);

    float weight = qualityPrime / (1.0 + quality);
    addPosition(vi[0], vpp[0], weight);
    addPosition(vi[1], vpp[1], weight);
    addPosition(vi[2], vpp[2], weight);
    addPosition(vi[3], vpp[3], weight);
    addPosition(vi[4], vpp[4], weight);
    addPosition(vi[5], vpp[5], weight);
    addPosition(vi[6], vpp[6], weight);
    addPosition(vi[7], vpp[7], weight);
}


__device__ smoothTetFct getmeSmoothTetPtr = getmeSmoothTet;
__device__ smoothPriFct getmeSmoothPriPtr = getmeSmoothPri;
__device__ smoothHexFct getmeSmoothHexPtr = getmeSmoothHex;


// CUDA Drivers
void installCudaGetmeSmoother()
{
    smoothTetFct d_smoothTet = nullptr;
    cudaMemcpyFromSymbol(&d_smoothTet, getmeSmoothTetPtr, sizeof(smoothTetFct));
    cudaMemcpyToSymbol(smoothTet, &d_smoothTet, sizeof(smoothTetFct));

    smoothPriFct d_smoothPri = nullptr;
    cudaMemcpyFromSymbol(&d_smoothPri, getmeSmoothPriPtr, sizeof(smoothPriFct));
    cudaMemcpyToSymbol(smoothPri, &d_smoothPri, sizeof(smoothPriFct));

    smoothHexFct d_smoothHex = nullptr;
    cudaMemcpyFromSymbol(&d_smoothHex, getmeSmoothHexPtr, sizeof(smoothHexFct));
    cudaMemcpyToSymbol(smoothHex, &d_smoothHex, sizeof(smoothHexFct));


    float h_lambda = 0.78;
    cudaMemcpyToSymbol(Lambda, &h_lambda, sizeof(float));


    if(verboseCuda)
        printf("I -> CUDA \tGETMe smoother installed\n");
}
