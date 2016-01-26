#include "../Discretizing/Base.cuh"


// Tetrahedron quality evaluation function
typedef float (*tetQualityFct)(const vec3 vp[TET_VERTEX_COUNT]);

extern __device__ tetQualityFct tetQualityImpl;


// Prism quality evaluation function
typedef float (*priQualityFct)(const vec3 vp[PRI_VERTEX_COUNT]);

extern __device__ priQualityFct priQualityImpl;


// Hexahedron quality evaluation function
typedef float (*hexQualityFct)(const vec3 vp[HEX_VERTEX_COUNT]);

extern __device__ hexQualityFct hexQualityImpl;
