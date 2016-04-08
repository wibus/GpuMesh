#include "Base.cuh"

__constant__ float NMValueConvergence;
__constant__ int NMSecurityCycleCount;
__constant__ float NMLocalSizeToNodeShift;
__constant__ float NMAlpha;
__constant__ float NMBeta;
__constant__ float NMGamma;
__constant__ float NMDelta;


// Smoothing Helper
__device__ float computeLocalElementSize(uint vId);
__device__ float patchQuality(uint vId);
__device__ void swap(vec4& v1, vec4& v2)
{
    glm::dvec4 tmp = v1;
    v1 = v2;
    v2 = tmp;
}

// ENTRY POINT //
__device__ void nelderMeadSmoothVert(uint vId)
{
    // Compute local element size
    float localSize = computeLocalElementSize(vId);

    // Initialize node shift distance
    float nodeShift = localSize * NMLocalSizeToNodeShift;


    Topo topo = topos[vId];
    vec3 pos = vec3(verts[vId].p);
    vec4 vo(pos, patchQuality(vId));

    vec4 simplex[TET_VERTEX_COUNT] = {
        vec4(pos + vec3(nodeShift, 0, 0), 0),
        vec4(pos + vec3(0, nodeShift, 0), 0),
        vec4(pos + vec3(0, 0, nodeShift), 0),
        vo
    };

    int cycle = 0;
    bool reset = false;
    bool terminated = false;
    while(!terminated)
    {
        for(uint p=0; p < TET_VERTEX_COUNT-1; ++p)
        {
            // Since 'pos' is a reference on vertex's position
            // modifing its value here should be seen by the evaluator
            if(topo.type > 0)
                verts[vId].p = vec4(snapToBoundary(topo.type, vec3(simplex[p])), 0);
            else
                verts[vId].p = simplex[p];

            // Compute patch quality
            simplex[p] = vec4(vec3(verts[vId].p), patchQuality(vId));
        }

        // Mini bubble sort
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);
        if(simplex[1].w > simplex[2].w)
            swap(simplex[1], simplex[2]);
        if(simplex[2].w > simplex[3].w)
            swap(simplex[2], simplex[3]);
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);
        if(simplex[1].w > simplex[2].w)
            swap(simplex[1], simplex[2]);
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);


        for(; cycle < NMSecurityCycleCount; ++cycle)
        {
            // Centroid
            vec3 c = 1/3.0f * (
                vec3(simplex[1]) +
                vec3(simplex[2]) +
                vec3(simplex[3]));

            float f = 0.0;

            // Reflect
            verts[vId].p = vec4(c + NMAlpha*(c - vec3(simplex[0])), 0);
            if(topo.type > 0) verts[vId].p = vec4(snapToBoundary(topo.type, vec3(verts[vId].p)), 0);
            float fr = f = patchQuality(vId);

            vec4 xr = verts[vId].p;

            // Expand
            if(simplex[3].w < fr)
            {
                verts[vId].p = vec4(c + NMGamma*(vec3(verts[vId].p) - c), 0);
                if(topo.type > 0) verts[vId].p = vec4(snapToBoundary(topo.type, vec3(verts[vId].p)), 0);
                float fe = f = patchQuality(vId);

                if(fe <= fr)
                {
                    verts[vId].p = xr;
                    f = fr;
                }
            }
            // Contract
            else if(simplex[1].w >= fr)
            {
                // Outside
                if(fr > simplex[0].w)
                {
                    verts[vId].p = vec4(c + NMBeta*(vec3(xr) - c), 0);
                    if(topo.type > 0) verts[vId].p = vec4(snapToBoundary(topo.type, vec3(verts[vId].p)), 0);
                    f = patchQuality(vId);
                }
                // Inside
                else
                {
                    verts[vId].p = vec4(c + NMBeta*(vec3(simplex[0]) - c), 0);
                    if(topo.type > 0) verts[vId].p = vec4(snapToBoundary(topo.type, vec3(verts[vId].p)), 0);
                    f = patchQuality(vId);
                }
            }

            // Insert new vertex in the working simplex
            vec4 vertex(vec3(verts[vId].p), f);
            if(vertex.w > simplex[3].w)
                swap(simplex[3], vertex);
            if(vertex.w > simplex[2].w)
                swap(simplex[2], vertex);
            if(vertex.w > simplex[1].w)
                swap(simplex[1], vertex);
            if(vertex.w > simplex[0].w)
                swap(simplex[0], vertex);


            if( (simplex[3].w - simplex[1].w) < NMValueConvergence )
            {
                terminated = true;
                break;
            }
        }

        if( terminated || (cycle >= NMSecurityCycleCount && reset) )
        {
            break;
        }
        else
        {
            simplex[0] = vo - vec4(nodeShift, 0, 0, 0);
            simplex[1] = vo - vec4(0, nodeShift, 0, 0);
            simplex[2] = vo - vec4(0, 0, nodeShift, 0);
            simplex[3] = vo;
            reset = true;
            cycle = 0;
        }
    }

    if(topo.type > 0)
        verts[vId].p = vec4(snapToBoundary(topo.type, vec3(simplex[3])), 0);
    else
        verts[vId].p = simplex[3];
}

__device__ smoothVertFct nelderMeadSmoothVertPtr = nelderMeadSmoothVert;


// CUDA Drivers
void installCudaNelderMeadSmoother(
        float h_valueConvergence,
        int h_securityCycleCount,
        float h_localSizeToNodeShift,
        float h_alpha,
        float h_beta,
        float h_gamma,
        float h_delta)
{
    smoothVertFct d_smoothVert = nullptr;
    cudaMemcpyFromSymbol(&d_smoothVert, nelderMeadSmoothVertPtr, sizeof(smoothVertFct));
    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));

    // TODO wbussiere 2016-04-04 : Pass security cycle count and
    //  local size to node shift from Smoother
    cudaMemcpyToSymbol(NMValueConvergence, &h_valueConvergence, sizeof(float));
    cudaMemcpyToSymbol(NMSecurityCycleCount, &h_securityCycleCount, sizeof(int));
    cudaMemcpyToSymbol(NMLocalSizeToNodeShift, &h_localSizeToNodeShift, sizeof(float));
    cudaMemcpyToSymbol(NMAlpha, &h_alpha, sizeof(float));
    cudaMemcpyToSymbol(NMBeta,  &h_beta,  sizeof(float));
    cudaMemcpyToSymbol(NMGamma, &h_gamma, sizeof(float));
    cudaMemcpyToSymbol(NMDelta, &h_delta, sizeof(float));


    if(verboseCuda)
        printf("I -> CUDA \tNelder Mead smoother installed\n");
}
