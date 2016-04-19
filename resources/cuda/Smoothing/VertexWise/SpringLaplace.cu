#include "Base.cuh"


// ENTRY POINT //
__device__ void springLaplaceSmoothVert(uint vId)
{
    vec3 patchCenter = computeVertexEquilibrium(vId);

    vec3 pos = verts[vId].p;
    pos = mix(pos, patchCenter, MoveCoeff);


    Topo topo = topos[vId];
    if(topo.type > 0)
    {
        pos = snapToBoundary(topo.type, pos);
    }


    // Write
    verts[vId].p = pos;
}

__device__ smoothVertFct springLaplaceSmoothVertPtr = springLaplaceSmoothVert;


// CUDA Drivers
void installCudaSpringLaplaceSmoother()
{
    smoothVertFct d_smoothVert = nullptr;
    cudaMemcpyFromSymbol(&d_smoothVert, springLaplaceSmoothVertPtr, sizeof(smoothVertFct));
    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));


    if(verboseCuda)
        printf("I -> CUDA \tSpring Laplace smoother installed\n");
}
