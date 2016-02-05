#include "../../Boundary/Base.cuh"
#include "../../Evaluating/Base.cuh"


// Tetrahedron smoothing function
typedef void (*smoothVertFct)(uint vId);
extern __device__ smoothVertFct smoothVert;


// Move coefficient to soften smoothing displacement
extern __device__  float MoveCoeff;
