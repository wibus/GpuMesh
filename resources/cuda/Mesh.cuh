#include <cuda.h>

#define GLM_FORCE_CUDA
#include <GLM/glm.hpp>


////////////////////
// Mesh tructures //
////////////////////

struct Vert
{
    glm::vec4 p;
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
};

struct Pri
{
    uint v[6];
};

struct Hex
{
    uint v[8];
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

// Topology indirection table
// type == -1 : fixed vertex
// type ==  0 : free vertex
// type  >  0 : boundary vertex
// When type > 0, type is boundary's ID
const int TOPO_FIXED = -1;
const int TOPO_FREE = 0;

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
__device__ Vert* verts;
__device__ Tet* tets;
__device__ Pri* pris;
__device__ Hex* hexs;
__device__ Topo* topos;
__device__ NeigVert* neigVerts;
__device__ NeigElem* neigElems;
__device__ uint* groupMembers;


const uint EVALUATE_QUALS_BUFFER_BINDING    = 8;
const uint VERTEX_ACCUMS_BUFFER_BINDING     = 9;
const uint KD_NODES_BUFFER_BINDING          = 10;
const uint KD_TETS_BUFFER_BINDING           = 11;
const uint KD_METRICS_BUFFER_BINDING        = 12;


//////////////////////////////////
// Mesh elements decompositions //
//////////////////////////////////

// Tetrahedron
const int TET_ELEMENT_TYPE = 0;

const uint TET_VERTEX_COUNT = 4;

const uint TET_EDGE_COUNT = 6;
__constant__ Edge TET_EDGES[TET_EDGE_COUNT];

const uint TET_TRI_COUNT = 4;
__constant__ Tri TET_TRIS[TET_TRI_COUNT];

const uint TET_TET_COUNT = 1;
__constant__ Tet TET_TETS[TET_TET_COUNT];


// Prism
const int PRI_ELEMENT_TYPE = 1;

const uint PRI_VERTEX_COUNT = 6;

const uint PRI_EDGE_COUNT = 9;
__constant__ Edge PRI_EDGES[PRI_EDGE_COUNT];

const uint PRI_TRI_COUNT = 8;
__constant__ Tri PRI_TRIS[PRI_TRI_COUNT];

const uint PRI_TET_COUNT = 3;
__constant__ Tet PRI_TETS[PRI_TET_COUNT];


// Hexahedron
const int HEX_ELEMENT_TYPE = 2;

const uint HEX_VERTEX_COUNT = 8;

const uint HEX_EDGE_COUNT = 12;
__constant__ Edge HEX_EDGES[HEX_EDGE_COUNT];

const uint HEX_TRI_COUNT = 12;
__constant__ Tri HEX_TRIS[HEX_TRI_COUNT];

const uint HEX_TET_COUNT = 5;
__constant__ Tet HEX_TETS[HEX_TET_COUNT];
