#include "DataStructures/GpuMesh.h"

#include "Mesh.cuh"



///////////////////////
// Mesh data buffers //
///////////////////////
__device__ Vert* verts;
__device__ uint verts_length;

__device__ Tet* tets;
__device__ uint tets_length;

__device__ Pri* pris;
__device__ uint pris_length;

__device__ Hex* hexs;
__device__ uint hexs_length;

__device__ Topo* topos;
__device__ uint topos_length;

__device__ NeigVert* neigVerts;
__device__ uint neigVerts_length;

__device__ NeigElem* neigElems;
__device__ uint neigElems_length;

__device__ uint* groupMembers;
__device__ uint groupMembers_length;


// CUDA Drivers
size_t d_tetCount = 0;
GpuTet* d_tets = nullptr;
void updateCudaTets(const std::vector<GpuTet>& tetBuff)
{
    uint tetLength = tetBuff.size();
    size_t tetBuffSize = sizeof(decltype(tetBuff.front())) * tetLength;
    if(d_tets == nullptr || d_tetCount != tetBuff.size())
    {
        cudaFree(d_tets);
        cudaMalloc(&d_tets, tetBuffSize);
        cudaMemcpyToSymbol(tets, &d_tets, sizeof(d_tets));

        cudaMemcpyToSymbol(tets_length, &tetLength, sizeof(uint));
    }

    cudaMemcpy(d_tets, tetBuff.data(), tetBuffSize, cudaMemcpyHostToDevice);
}


size_t d_priCount = 0;
GpuPri* d_pris = nullptr;
void updateCudaPris(const std::vector<GpuPri>& priBuff)
{
    uint priLength = priBuff.size();
    size_t priBuffSize = sizeof(decltype(priBuff.front())) * priLength;
    if(d_pris == nullptr || d_priCount != priBuff.size())
    {
        cudaFree(d_pris);
        cudaMalloc(&d_pris, priBuffSize);
        cudaMemcpyToSymbol(pris, &d_pris, sizeof(d_pris));

        cudaMemcpyToSymbol(pris_length, &priLength, sizeof(uint));
    }

    cudaMemcpy(d_pris, priBuff.data(), priBuffSize, cudaMemcpyHostToDevice);
}


size_t d_hexCount = 0;
GpuHex* d_hexs = nullptr;
void updateCudaHexs(const std::vector<GpuHex>& hexBuff)
{
    uint hexLength = hexBuff.size();
    size_t hexBuffSize = sizeof(decltype(hexBuff.front())) * hexLength;
    if(d_hexs == nullptr || d_hexCount != hexBuff.size())
    {
        cudaFree(d_hexs);
        cudaMalloc(&d_hexs, hexBuffSize);
        cudaMemcpyToSymbol(hexs, &d_hexs, sizeof(d_hexs));

        cudaMemcpyToSymbol(hexs_length, &hexLength, sizeof(uint));
    }

    cudaMemcpy(d_hexs, hexBuff.data(), hexBuffSize, cudaMemcpyHostToDevice);
}


size_t d_topoCount = 0;
Topo* d_topos = nullptr;

size_t d_neighVertCount = 0;
NeigVert* d_neigVerts = nullptr;

size_t d_neighElemCount = 0;
NeigElem* d_neigElems = nullptr;

void updateCudaTopo(
        const std::vector<GpuTopo>& topoBuff,
        const std::vector<GpuNeigVert>& neigVertBuff,
        const std::vector<GpuNeigElem>& neigElemBuff)
{
    // Topologies
    uint topoLength = topoBuff.size();
    size_t topoBuffSize = sizeof(decltype(topoBuff.front())) * topoLength;
    if(d_topos == nullptr || d_topoCount != topoBuff.size())
    {
        cudaFree(d_topos);
        cudaMalloc(&d_topos, topoBuffSize);
        cudaMemcpyToSymbol(topos, &d_topos, sizeof(d_topos));

        cudaMemcpyToSymbol(topos_length, &topoLength, sizeof(uint));
    }

    cudaMemcpy(d_topos, topoBuff.data(), topoBuffSize, cudaMemcpyHostToDevice);


    // Neighbor vertices
    uint neigVertLength = neigVertBuff.size();
    size_t neigVertBuffSize = sizeof(decltype(neigVertBuff.front())) * neigVertLength;
    if(d_neigVerts == nullptr || d_neighVertCount != neigVertBuff.size())
    {
        cudaFree(d_neigVerts);
        cudaMalloc(&d_neigVerts, neigVertBuffSize);
        cudaMemcpyToSymbol(neigVerts, &d_neigVerts, sizeof(d_neigVerts));

        cudaMemcpyToSymbol(neigVerts_length, &neigVertLength, sizeof(uint));
    }

    cudaMemcpy(d_neigVerts, neigVertBuff.data(), neigVertBuffSize, cudaMemcpyHostToDevice);


    // Neighbor elements
    uint neigElemLength = neigElemBuff.size();
    size_t neigElemBuffSize = sizeof(decltype(neigElemBuff.front())) * neigElemLength;
    if(d_neigElems == nullptr || d_neighElemCount != neigElemBuff.size())
    {
        cudaFree(d_neigElems);
        cudaMalloc(&d_neigElems, neigElemBuffSize);
        cudaMemcpyToSymbol(neigElems, &d_neigElems, sizeof(d_neigElems));

        cudaMemcpyToSymbol(neigElems_length, &neigElemLength, sizeof(uint));
    }

    cudaMemcpy(d_neigElems, neigElemBuff.data(), neigElemBuffSize, cudaMemcpyHostToDevice);
}


size_t d_groupMembersCount = 0;
GLuint* d_groupMembers = nullptr;
void updateCudaGroupMembers(
        const std::vector<GLuint>& groupMemberBuff)
{
    // Group members
    uint groupMembersLength = groupMemberBuff.size();
    size_t groupMembersBuffSize = sizeof(decltype(groupMemberBuff.front())) * groupMembersLength;
    if(d_groupMembers == nullptr || d_groupMembersCount != groupMemberBuff.size())
    {
        cudaFree(d_groupMembers);
        cudaMalloc(&d_groupMembers, groupMembersBuffSize);
        cudaMemcpyToSymbol(groupMembers, &d_groupMembers, sizeof(d_groupMembers));

        cudaMemcpyToSymbol(groupMembers_length, &groupMembersLength, sizeof(uint));
    }

    cudaMemcpy(d_groupMembers, groupMemberBuff.data(), groupMembersBuffSize, cudaMemcpyHostToDevice);
}
