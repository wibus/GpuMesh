#include "Base.cuh"

__constant__ int GDSecurityCycleCount;
__constant__ float GDLocalSizeToNodeShift;


// Smoothing Helper
__device__ float computeLocalElementSize(uint vId);
__device__ float patchQuality(uint vId);


// ENTRY POINT //
__device__ void gradientDescentSmoothVert(uint vId)
{
    // Compute local element size
    float localSize = computeLocalElementSize(vId);

    // Initialize node shift distance
    float nodeShift = localSize * GDLocalSizeToNodeShift;
    float originalNodeShift = nodeShift;

    for(int c=0; c < GDSecurityCycleCount; ++c)
    {
        // Define patch quality gradient samples
        vec3 pos = vec3(verts[vId].p);
        const uint GRADIENT_SAMPLE_COUNT = 6;
        float sampleQualities[GRADIENT_SAMPLE_COUNT] =
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        vec3 gradSamples[GRADIENT_SAMPLE_COUNT] = {
            pos + vec3(-nodeShift, 0.0,   0.0),
            pos + vec3( nodeShift, 0.0,   0.0),
            pos + vec3( 0.0,  -nodeShift, 0.0),
            pos + vec3( 0.0,   nodeShift, 0.0),
            pos + vec3( 0.0,   0.0,  -nodeShift),
            pos + vec3( 0.0,   0.0,   nodeShift)
        };

        Topo topo = topos[vId];
        if(topo.type > 0)
        {
            for(uint p=0; p < GRADIENT_SAMPLE_COUNT; ++p)
                gradSamples[p] = snapToBoundary(
                    topo.type, gradSamples[p]);
        }

        for(uint p=0; p < GRADIENT_SAMPLE_COUNT; ++p)
        {
            // Quality evaluation functions will use this updated position
            // to compute element shape measures.
            verts[vId].p = vec4(gradSamples[p], 0.0);

            // Compute patch quality
            sampleQualities[p] = patchQuality(vId);
        }
        verts[vId].p = vec4(pos, 0.0);

        vec3 gradQ = vec3(
            sampleQualities[1] - sampleQualities[0],
            sampleQualities[3] - sampleQualities[2],
            sampleQualities[5] - sampleQualities[4]);
        float gradQNorm = length(gradQ);

        if(gradQNorm == 0)
            break;


        const uint PROPOSITION_COUNT = 7;
        const float OFFSETS[PROPOSITION_COUNT] = {
            -0.25,
             0.00,
             0.25,
             0.50,
             0.75,
             1.00,
             1.25
        };

        vec3 shift = gradQ * (nodeShift / gradQNorm);
        vec3 propositions[PROPOSITION_COUNT] = {
            pos + shift * OFFSETS[0],
            pos + shift * OFFSETS[1],
            pos + shift * OFFSETS[2],
            pos + shift * OFFSETS[3],
            pos + shift * OFFSETS[4],
            pos + shift * OFFSETS[5],
            pos + shift * OFFSETS[6]
        };

        if(topo.type > 0)
        {
            for(uint p=0; p < PROPOSITION_COUNT; ++p)
                propositions[p] = snapToBoundary(
                    topo.type, propositions[p]);
        }

        uint bestProposition = 0;
        float bestQualityMean = -1.0/0.0; // -Inf
        for(uint p=0; p < PROPOSITION_COUNT; ++p)
        {
            // Quality evaluation functions will use this updated position
            // to compute element shape measures.
            verts[vId].p = vec4(propositions[p], 0.0);

            // Compute patch quality
            float pq = patchQuality(vId);

            if(pq > bestQualityMean)
            {
                bestQualityMean = pq;
                bestProposition = p;
            }
        }


        // Update vertex's position
        verts[vId].p = vec4(propositions[bestProposition], 0.0);

        // Scale node shift and stop if it is too small
        nodeShift *= abs(OFFSETS[bestProposition]);
        if(nodeShift < originalNodeShift / 10.0)
            break;
    }
}

__device__ smoothVertFct gradientDescentSmoothVertPtr = gradientDescentSmoothVert;


// CUDA Drivers
void installCudaGradientDescentSmoother(
        int h_securityCycleCount,
        float h_localSizeToNodeShift)
{
    smoothVertFct d_smoothVert = nullptr;
    cudaMemcpyFromSymbol(&d_smoothVert, gradientDescentSmoothVertPtr, sizeof(smoothVertFct));
    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));

    // TODO wbussiere 2016-04-04 : Pass security cycle count and
    //  local size to node shift from Smoother

    cudaMemcpyToSymbol(GDSecurityCycleCount, &h_securityCycleCount, sizeof(int));
    cudaMemcpyToSymbol(GDLocalSizeToNodeShift, &h_localSizeToNodeShift, sizeof(float));

    printf("I -> CUDA \tGradient Descent smoother installed\n");
}
