#include "Base.cuh"


// ENTRY POINT //
__device__ void springLaplaceSmoothVert(uint vId)
{
    vec3 patchCenter = computeVertexEquilibrium(vId);

    vec3 pos = vec3(verts[vId].p);
    pos = mix(pos, patchCenter, MoveCoeff);


    Topo topo = topos[vId];
    if(topo.type > 0)
    {
        pos = snapToBoundary(topo.type, pos);
    }


    // Write
    verts[vId].p = vec4(pos, 0.0);
}

__device__ smoothVertFct springLaplaceSmoothVertPtr = springLaplaceSmoothVert;


// CUDA Drivers
void installCudaSpringLaplaceSmoother()
{
    smoothVertFct d_smoothVert = nullptr;
    cudaMemcpyFromSymbol(&d_smoothVert, springLaplaceSmoothVertPtr, sizeof(smoothVertFct));
    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));

    printf("I -> CUDA \tSpring Laplace smoother installed\n");
}
