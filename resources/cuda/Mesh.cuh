#include <cstdio>

#include <cuda.h>

#include <GLM/glm.hpp>
using namespace glm;


//////////////////////////////////
// Mesh elements decompositions //
//////////////////////////////////

// Tetrahedron
#define TET_ELEMENT_TYPE int(0)
#define TET_VERTEX_COUNT uint(4)


// Pyramid
#define PYR_ELEMENT_TYPE int(1)
#define PYR_VERTEX_COUNT uint(5)


// Prism
#define PRI_ELEMENT_TYPE int(2)
#define PRI_VERTEX_COUNT uint(6)


// Hexahedron
#define HEX_ELEMENT_TYPE int(3)
#define HEX_VERTEX_COUNT uint(8)



////////////////////
// Mesh tructures //
////////////////////

struct Vert
{
    vec3 p;
    mutable uint c;
};

struct Edge
{
    uint v[2];
};

struct Tri
{
    uint v[3];
};

struct Tet
{
    uint v[4];
    mutable uint c[1];
};

struct Pri
{
    uint v[6];
    mutable uint c[6];
};

struct Hex
{
    uint v[8];
    mutable uint c[8];
};

struct NeigVert
{
    uint v;
};

struct NeigElem
{
    int type;
    uint id;
};

struct PatchElem
{
    __device__ PatchElem() {}

    uint type;
    uint n;

    Tet tet;
    Pri pri;
    Hex hex;
    vec3 p[HEX_VERTEX_COUNT];
};


// Topology indirection table
// type == -1 : fixed vertex
// type ==  0 : free vertex
// type  >  0 : boundary vertex
// When type > 0, type is boundary's ID
#define TOPO_FIXED -1
#define TOPO_FREE   0

struct Topo
{
    int type;
    uint neigVertBase;
    uint neigVertCount;
    uint neigElemBase;
    uint neigElemCount;
};

///////////////////////
// Mesh data buffers //
///////////////////////
extern __constant__ uint verts_length;
extern __device__ Vert* verts;

extern __constant__ uint tets_length;
extern __device__ Tet* tets;

extern __constant__ uint pris_length;
extern __device__ Pri* pris;

extern __constant__ uint hexs_length;
extern __device__ Hex* hexs;

extern __constant__ uint topos_length;
extern __device__ Topo* topos;

extern __constant__ uint neigVerts_length;
extern __device__ NeigVert* neigVerts;

extern __constant__ uint neigElems_length;
extern __device__ NeigElem* neigElems;

extern __constant__ uint groupMembers_length;
extern __device__ uint* groupMembers;

// Reference mesh vertices
extern __constant__ uint refVerts_length;
extern __device__ Vert* refVerts;

// Reference mesh metrics
extern __constant__ uint refMetrics_length;
extern __device__ mat4* refMetrics;

// GPU independent groups
extern __constant__ int GroupBase;
extern __constant__ int GroupSize;




/////////////////////////
// CUDA Error Handling //
/////////////////////////
extern bool verboseCuda;

#ifndef NDEBUG
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)
#else
#define cudaCheckErrors(msg)
#endif
