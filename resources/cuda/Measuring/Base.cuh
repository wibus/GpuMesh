#include "../Sampling/Base.cuh"


// Riemannian distance
typedef float (*riemannianDistanceFct)(const vec3& a, const vec3& b);
extern __device__ riemannianDistanceFct riemannianDistanceImpl;


// Riemannian segment
typedef vec3 (*riemannianSegmentFct)(const vec3& a, const vec3& b);
extern __device__ riemannianSegmentFct riemannianSegmentImpl;



// Tetrahedron Volume evaluation function
typedef float (*tetVolumeFct)(const vec3 vp[TET_VERTEX_COUNT]);
extern __device__ tetVolumeFct tetVolumeImpl;


// Prism Volume evaluation function
typedef float (*priVolumeFct)(const vec3 vp[PRI_VERTEX_COUNT]);
extern __device__ priVolumeFct priVolumeImpl;


// Hexahedron Volume evaluation function
typedef float (*hexVolumeFct)(const vec3 vp[HEX_VERTEX_COUNT]);
extern __device__ hexVolumeFct hexVolumeImpl;



// Compute Vertex Equilibrium
typedef vec3 (*computeVertexEquilibriumFct)(uint vId);
extern __device__ computeVertexEquilibriumFct computeVertexEquilibrium;
