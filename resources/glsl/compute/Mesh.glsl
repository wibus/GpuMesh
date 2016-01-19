#version 450

////////////////
// Extensions //
////////////////
#extension GL_NV_shader_atomic_float : enable


////////////////////
// Mesh tructures //
////////////////////

struct Vert
{
    vec4 p;
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
layout(std140, binding = 0) buffer Verts
{
    Vert verts[];
};

layout(shared, binding = 1) buffer Tets
{
    Tet tets[];
};

layout(shared, binding = 2) buffer Pris
{
    Pri pris[];
};

layout(shared, binding = 3) buffer Hexs
{
    Hex hexs[];
};

layout(shared, binding = 4) buffer Topos
{
    Topo topos[];
};

layout(shared, binding = 5) buffer NeigVerts
{
    NeigVert neigVerts[];
};

layout(shared, binding = 6) buffer NeigElems
{
    NeigElem neigElems[];
};

layout(shared, binding = 7) buffer GroupMembers
{
    uint groupMembers[];
};


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
uniform Edge TET_EDGES[TET_EDGE_COUNT];

const uint TET_TRI_COUNT = 4;
uniform Tri TET_TRIS[TET_TRI_COUNT];

const uint TET_TET_COUNT = 1;
uniform Tet TET_TETS[TET_TET_COUNT];


// Prism
const int PRI_ELEMENT_TYPE = 1;

const uint PRI_VERTEX_COUNT = 6;

const uint PRI_EDGE_COUNT = 9;
uniform Edge PRI_EDGES[PRI_EDGE_COUNT];

const uint PRI_TRI_COUNT = 8;
uniform Tri PRI_TRIS[PRI_TRI_COUNT];

const uint PRI_TET_COUNT = 3;
uniform Tet PRI_TETS[PRI_TET_COUNT];


// Hexahedron
const int HEX_ELEMENT_TYPE = 2;

const uint HEX_VERTEX_COUNT = 8;

const uint HEX_EDGE_COUNT = 12;
uniform Edge HEX_EDGES[HEX_EDGE_COUNT];

const uint HEX_TRI_COUNT = 12;
uniform Tri HEX_TRIS[HEX_TRI_COUNT];

const uint HEX_TET_COUNT = 5;
uniform Tet HEX_TETS[HEX_TET_COUNT];
