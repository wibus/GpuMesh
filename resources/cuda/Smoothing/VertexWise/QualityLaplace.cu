#include "Base.cuh"

__device__  float QLMoveCoeff = 0.35;


// Smoothing helper
__device__ float patchQuality(uint vId);


// ENTRY POINT //
__device__ void qualityLaplaceSmoothVert(uint vId)
{
    // Compute patch center
    vec3 pos = verts[vId].p;
    vec3 patchCenter = computeVertexEquilibrium(vId);
    vec3 centerDist = patchCenter - pos;

    const uint PROPOSITION_COUNT = 8;
    const float OFFSETS[PROPOSITION_COUNT] = {
        -0.25, 0.00, 0.10, 0.20,
         0.40, 0.80, 1.20, 1.60
    };

    // Define propositions for new vertex's position
    vec3 shift = centerDist * QLMoveCoeff;
    vec3 propositions[PROPOSITION_COUNT] = {
        pos + shift * OFFSETS[0],
        pos + shift * OFFSETS[1],
        pos + shift * OFFSETS[2],
        pos + shift * OFFSETS[3],
        pos + shift * OFFSETS[4],
        pos + shift * OFFSETS[5],
        pos + shift * OFFSETS[6],
        pos + shift * OFFSETS[7]
    };


    // Choose best position based on quality geometric mean
    uint bestProposition = 0;
    float bestQualityMean = -1.0/0.0; // -Inf
    for(uint p=0; p < PROPOSITION_COUNT; ++p)
    {
        // Quality evaluation functions will use this updated position
        // to compute element shape measures.
        verts[vId].p = propositions[p];

        float pq = patchQuality(vId);

        if(pq > bestQualityMean)
        {
            bestQualityMean = pq;
            bestProposition = p;
        }
    }


    // Update vertex's position
    verts[vId].p = propositions[bestProposition];
}

__device__ smoothVertFct qualityLaplaceSmoothVertPtr = qualityLaplaceSmoothVert;


// CUDA Drivers
void installCudaQualityLaplaceSmoother(float moveCoeff)
{
    cudaMemcpyToSymbol(QLMoveCoeff, &moveCoeff, sizeof(float));

    smoothVertFct d_smoothVert = nullptr;
    cudaMemcpyFromSymbol(&d_smoothVert, qualityLaplaceSmoothVertPtr, sizeof(smoothVertFct));
    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));


    if(verboseCuda)
        printf("I -> CUDA \tQuality Laplace smoother installed\n");
}
