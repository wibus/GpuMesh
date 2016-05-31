#include "Base.cuh"

#define PIPE_CYLINDER_FACE_ID 1
#define PIPE_YNEG_DISK_FACE_ID 2
#define PIPE_YPOS_DISK_FACE_ID 3
#define PIPE_YNEG_CIRCLE_EDGE_ID 4
#define PIPE_YPOS_CIRCLE_EDGE_ID 5

#define PIPE_RADIUS 0.3f
#define EXT_NORMAL vec3(1.0, 0, 0)
#define EXT_CENTER vec3(-1, 0.5, 0.0)


__device__ vec3 snapToPipeSurface(vec3 pos)
{
    vec3 center;

    if(pos.x < 0.5) // Straights
    {
        center = vec3(pos.x, (pos.y < 0.0 ? -0.5 : 0.5), 0.0);
    }
    else // Arc
    {
        center = pos - vec3(0.5, 0.0, pos.z);
        center = normalize(center) * 0.5f;
        center = vec3(0.5, 0, 0) + center;
    }

    vec3 dist = pos - center;
    vec3 extProj = normalize(dist) * PIPE_RADIUS;
    return center + extProj;
}

__device__ vec3 snapToPipeExtremityFace(vec3 pos)
{
    vec3 center = EXT_CENTER;
    center.y *= sign(pos.y);

    float offset = dot(pos - center, EXT_NORMAL);
    return pos - EXT_NORMAL * offset;
}

__device__ vec3 snapToPipeExtremityEdge(vec3 pos)
{
    vec3 center = EXT_CENTER;
    center.y *= sign(pos.y);

    vec3 dist = pos - center;
    float offset = dot(dist, EXT_NORMAL);
    vec3 extProj = dist - EXT_NORMAL * offset;
    return center + normalize(extProj) * PIPE_RADIUS;
}

__device__ vec3 pipeSnapToBoundary(int boundaryID, vec3 pos)
{
    switch(boundaryID)
    {
    case PIPE_CYLINDER_FACE_ID :
        return snapToPipeSurface(pos);
    case PIPE_YNEG_DISK_FACE_ID :
    case PIPE_YPOS_DISK_FACE_ID :
        return snapToPipeExtremityFace(pos);
    case PIPE_YNEG_CIRCLE_EDGE_ID :
    case PIPE_YPOS_CIRCLE_EDGE_ID :
        return snapToPipeExtremityEdge(pos);
    }

    return pos;
}


__device__ snapToBoundaryFct pipeSnapToBoundaryPtr = pipeSnapToBoundary;


// CUDA Drivers
void installCudaPipeBoundary()
{
    snapToBoundaryFct d_snapToBoundary = nullptr;
    cudaMemcpyFromSymbol(&d_snapToBoundary, pipeSnapToBoundaryPtr, sizeof(snapToBoundaryFct));
    cudaMemcpyToSymbol(snapToBoundary, &d_snapToBoundary, sizeof(snapToBoundaryFct));


    if(verboseCuda)
        printf("I -> CUDA \tPipe boundary installed\n");
}
