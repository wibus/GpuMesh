#version 440

////////////////////
// Mesh tructures //
////////////////////

struct Vert
{
    vec4 p;
};

struct Edge
{
    int v[2];
};

struct Tri
{
    int v[3];
};

struct Tet
{
    int v[4];
};

struct Pri
{
    int v[6];
};

struct Hex
{
    int v[8];
};

// Topology indirection table
// type == -1 : fixed vertex
// type ==  0 : free vertex
// type  >  0 : boundary vertex
// When type > 0, type is boundary's ID
struct Topo
{
    int type;
    int neigVertBase;
    int neigVertCount;
    int neigElemBase;
    int neigElemCount;
};

struct NeigVert
{
    int v;
};

struct NeigElem
{
    int type;
    int id;
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

const int FIRST_FREE_BUFFER_BINDING = 7;

//////////////////////////////////
// Mesh elements decompositions //
//////////////////////////////////

// Tetrahedron
const int TET_ELEMENT_TYPE = 0;

const int TET_EDGE_COUNT = 6;
uniform Edge TET_EDGES[TET_EDGE_COUNT];

const int TET_TRI_COUNT = 4;
uniform Tri TET_TRIS[TET_TRI_COUNT];

const int TET_TET_COUNT = 1;
uniform Tet TET_TETS[TET_TET_COUNT];


// Prism
const int PRI_ELEMENT_TYPE = 1;

const int PRI_EDGE_COUNT = 9;
uniform Edge PRI_EDGES[PRI_EDGE_COUNT];

const int PRI_TRI_COUNT = 8;
uniform Tri PRI_TRIS[PRI_TRI_COUNT];

const int PRI_TET_COUNT = 3;
uniform Tet PRI_TETS[PRI_TET_COUNT];


// Hexahedron
const int HEX_ELEMENT_TYPE = 2;

const int HEX_EDGE_COUNT = 12;
uniform Edge HEX_EDGES[HEX_EDGE_COUNT];

const int HEX_TRI_COUNT = 12;
uniform Tri HEX_TRIS[HEX_TRI_COUNT];

const int HEX_TET_COUNT = 5;
uniform Tet HEX_TETS[HEX_TET_COUNT];
