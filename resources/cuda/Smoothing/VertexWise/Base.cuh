#include "../../Evaluating/Base.cuh"


// Tetrahedron smoothing function
typedef void (*smoothVertFct)(uint vId);
extern __device__ smoothVertFct smoothVert;
