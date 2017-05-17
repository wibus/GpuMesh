#version 450


///////////////
// Constants //
///////////////
#define M_PI 3.1415926535897932384626433832795


////////////////
// Extensions //
////////////////
#extension GL_NV_shader_atomic_float : enable
#extension GL_ARB_compute_variable_group_size : enable

layout(local_size_variable) in;


//////////////////////////////////
// Mesh elements decompositions //
//////////////////////////////////

// Tetrahedron
const int TET_ELEMENT_TYPE = 0;
const uint TET_VERTEX_COUNT = 4;


// Pyramid
const int PYR_ELEMENT_TYPE = 1;
const uint PYR_VERTEX_COUNT = 5;


// Prism
const int PRI_ELEMENT_TYPE = 2;
const uint PRI_VERTEX_COUNT = 6;


// Hexahedron
const int HEX_ELEMENT_TYPE = 3;
const uint HEX_VERTEX_COUNT = 8;


// Vertex count in parameters
// We need a common array length to pass vertex positions to mesurement and
// evaluation functions in GLSL as the language does not allow us to pass
// arrays of undefined size or pointers as function paramters. So we chose the
// largest array size (hexaedron's) as th template container for vertices.
const int PARAM_VERTEX_COUNT = 8;



////////////////////
// Mesh tructures //
////////////////////

struct Vert
{
    vec3 p;
    uint c;
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
    uint c[1];
};

struct Pri
{
    uint v[6];
    uint c[6];
};

struct Hex
{
    uint v[8];
    uint c[8];
};

struct NeigVert
{
    uint v;
};

struct NeigElem
{
    uint id;
    int type;
    int vId;
};

struct PatchElem
{
    uint n;
    uint type;

    Tet tet; Pri pri; Hex hex;
    vec3 p[PARAM_VERTEX_COUNT];
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

const uint EVALUATE_QUAL_BUFFER_BINDING     = 8;
const uint EVALUATE_HIST_BUFFER_BINDING     = 9;
const uint VERTEX_ACCUMS_BUFFER_BINDING     = 10;
const uint REF_VERTS_BUFFER_BINDING         = 11;
const uint REF_METRICS_BUFFER_BINDING       = 12;
const uint KD_NODES_BUFFER_BINDING          = 14;
const uint LOCAL_TETS_BUFFER_BINDING        = 15;
const uint SPAWN_OFFSETS_BUFFER_BINDING     = 16;

const uint METRIC_AT_SUBROUTINE_LOC         = 0;
const uint METRIC_AT_SUBROUTINE_IDX         = 0;
const uint PATCH_QUALITY_SUBROUTINE_LOC     = 1;
const uint PATCH_QUALITY_SUBROUTINE_IDX     = 1;
const uint PARALLEL_PATCH_QUALITY_SUBROUTINE_LOC = 2;
const uint PARALLEL_PATCH_QUALITY_SUBROUTINE_IDX = 2;


layout(shared, binding = 0) buffer Verts
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

layout(std140, binding = REF_VERTS_BUFFER_BINDING) buffer RefVerts
{
    Vert refVerts[];
};

layout(std140, binding = REF_METRICS_BUFFER_BINDING) buffer RefMetrics
{
    mat4 refMetrics[];
};

Tri MeshTet_tris[] = {
    {{1, 2, 3}},
    {{0, 3, 2}},
    {{0, 1, 3}},
    {{0, 2, 1}}};
