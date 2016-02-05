#include <cstdio>

#include <GLM/glm.hpp>
using namespace glm;


// Boundary snap function
typedef vec3 (*snapToBoundaryFct)(int boundaryID, vec3 pos);
extern __device__ snapToBoundaryFct snapToBoundary;
