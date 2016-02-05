#include "../../Boundary/Base.cuh"
#include "../../Evaluating/Base.cuh"


// Tetrahedron smoothing function
typedef void (*smoothTetFct)(uint eId);
extern __device__ smoothTetFct smoothTet;


// Prism smoothing function
typedef void (*smoothPriFct)(uint eId);
extern __device__ smoothPriFct smoothPri;


// Hexahedron smoothing function
typedef void (*smoothHexFct)(uint eId);
extern __device__ smoothHexFct smoothHex;
