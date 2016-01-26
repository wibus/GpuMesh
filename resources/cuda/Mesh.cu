#include "DataStructures/GpuMesh.h"

#include "Mesh.cuh"



///////////////////////
// Mesh data buffers //
///////////////////////
__device__ Vert* verts = nullptr;
__device__ uint verts_length = 0;

__device__ Tet* tets = nullptr;
__device__ uint tets_length = 0;

__device__ Pri* pris = nullptr;
__device__ uint pris_length = 0;

__device__ Hex* hexs = nullptr;
__device__ uint hexs_length = 0;

__device__ Topo* topos = nullptr;
__device__ uint topos_length = 0;

__device__ NeigVert* neigVerts = nullptr;
__device__ uint neigVerts_length = 0;

__device__ NeigElem* neigElems = nullptr;
__device__ uint neigElems_length = 0;

__device__ uint* groupMembers = nullptr;
__device__ uint groupMembers_length = 0;


// CUDA Drivers
size_t d_vertsLength = 0;
GpuVert* d_verts = nullptr;
void updateCudaVerts(const GpuVert* vertsBuff, size_t vertsLength)
{
    size_t vertsBuffSize = sizeof(GpuVert) * vertsLength;
    if(d_verts == nullptr || d_vertsLength != vertsLength)
    {
        cudaFree(d_verts);
        cudaMalloc(&d_verts, vertsBuffSize);
        cudaMemcpyToSymbol(verts, &d_verts, sizeof(d_verts));

        d_vertsLength = vertsLength;
        cudaMemcpyToSymbol(verts_length, &vertsLength, sizeof(uint));
    }

    cudaMemcpy(d_verts, vertsBuff, vertsBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tverts updated\n");
}


size_t d_tetLength = 0;
GpuTet* d_tets = nullptr;
void updateCudaTets(const std::vector<GpuTet>& tetBuff)
{
    uint tetLength = tetBuff.size();
    size_t tetBuffSize = sizeof(decltype(tetBuff.front())) * tetLength;
    if(d_tets == nullptr || d_tetLength != tetLength)
    {
        cudaFree(d_tets);
        cudaMalloc(&d_tets, tetBuffSize);
        cudaMemcpyToSymbol(tets, &d_tets, sizeof(d_tets));

        d_tetLength = tetLength;
        cudaMemcpyToSymbol(tets_length, &tetLength, sizeof(uint));
    }

    cudaMemcpy(d_tets, tetBuff.data(), tetBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \ttets updated\n");
}


size_t d_priLength = 0;
GpuPri* d_pris = nullptr;
void updateCudaPris(const std::vector<GpuPri>& priBuff)
{
    uint priLength = priBuff.size();
    size_t priBuffSize = sizeof(decltype(priBuff.front())) * priLength;
    if(d_pris == nullptr || d_priLength != priLength)
    {
        cudaFree(d_pris);
        cudaMalloc(&d_pris, priBuffSize);
        cudaMemcpyToSymbol(pris, &d_pris, sizeof(d_pris));

        d_priLength = priLength;
        cudaMemcpyToSymbol(pris_length, &priLength, sizeof(uint));
    }

    cudaMemcpy(d_pris, priBuff.data(), priBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tpris updated\n");
}


size_t d_hexLength = 0;
GpuHex* d_hexs = nullptr;
void updateCudaHexs(const std::vector<GpuHex>& hexBuff)
{
    uint hexLength = hexBuff.size();
    size_t hexBuffSize = sizeof(decltype(hexBuff.front())) * hexLength;
    if(d_hexs == nullptr || d_hexLength != hexLength)
    {
        cudaFree(d_hexs);
        cudaMalloc(&d_hexs, hexBuffSize);
        cudaMemcpyToSymbol(hexs, &d_hexs, sizeof(d_hexs));

        d_hexLength = hexLength;
        cudaMemcpyToSymbol(hexs_length, &hexLength, sizeof(uint));
    }

    cudaMemcpy(d_hexs, hexBuff.data(), hexBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \thexs updated\n");
}


size_t d_topoLength = 0;
Topo* d_topos = nullptr;

size_t d_neighVertLength = 0;
NeigVert* d_neigVerts = nullptr;

size_t d_neighElemLength = 0;
NeigElem* d_neigElems = nullptr;

void updateCudaTopo(
        const std::vector<GpuTopo>& topoBuff,
        const std::vector<GpuNeigVert>& neigVertBuff,
        const std::vector<GpuNeigElem>& neigElemBuff)
{
    // Topologies
    uint topoLength = topoBuff.size();
    size_t topoBuffSize = sizeof(decltype(topoBuff.front())) * topoLength;
    if(d_topos == nullptr || d_topoLength != topoLength)
    {
        cudaFree(d_topos);
        cudaMalloc(&d_topos, topoBuffSize);
        cudaMemcpyToSymbol(topos, &d_topos, sizeof(d_topos));

        d_topoLength = topoLength;
        cudaMemcpyToSymbol(topos_length, &topoLength, sizeof(uint));
    }

    cudaMemcpy(d_topos, topoBuff.data(), topoBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \ttopos updated\n");


    // Neighbor vertices
    uint neigVertLength = neigVertBuff.size();
    size_t neigVertBuffSize = sizeof(decltype(neigVertBuff.front())) * neigVertLength;
    if(d_neigVerts == nullptr || d_neighVertLength != neigVertLength)
    {
        cudaFree(d_neigVerts);
        cudaMalloc(&d_neigVerts, neigVertBuffSize);
        cudaMemcpyToSymbol(neigVerts, &d_neigVerts, sizeof(d_neigVerts));

        d_neighVertLength = neigVertLength;
        cudaMemcpyToSymbol(neigVerts_length, &neigVertLength, sizeof(uint));
    }

    cudaMemcpy(d_neigVerts, neigVertBuff.data(), neigVertBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tneigVerts updated\n");


    // Neighbor elements
    uint neigElemLength = neigElemBuff.size();
    size_t neigElemBuffSize = sizeof(decltype(neigElemBuff.front())) * neigElemLength;
    if(d_neigElems == nullptr || d_neighElemLength != neigElemLength)
    {
        cudaFree(d_neigElems);
        cudaMalloc(&d_neigElems, neigElemBuffSize);
        cudaMemcpyToSymbol(neigElems, &d_neigElems, sizeof(d_neigElems));

        d_neighElemLength = neigElemLength;
        cudaMemcpyToSymbol(neigElems_length, &neigElemLength, sizeof(uint));
    }

    cudaMemcpy(d_neigElems, neigElemBuff.data(), neigElemBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tneigElems updated\n");
}


size_t d_groupMembersLength = 0;
GLuint* d_groupMembers = nullptr;
void updateCudaGroupMembers(
        const std::vector<GLuint>& groupMemberBuff)
{
    // Group members
    uint groupMembersLength = groupMemberBuff.size();
    size_t groupMembersBuffSize = sizeof(decltype(groupMemberBuff.front())) * groupMembersLength;
    if(d_groupMembers == nullptr || d_groupMembersLength != groupMembersLength)
    {
        cudaFree(d_groupMembers);
        cudaMalloc(&d_groupMembers, groupMembersBuffSize);
        cudaMemcpyToSymbol(groupMembers, &d_groupMembers, sizeof(d_groupMembers));

        d_groupMembersLength = groupMembersLength;
        cudaMemcpyToSymbol(groupMembers_length, &groupMembersLength, sizeof(uint));
    }

    cudaMemcpy(d_groupMembers, groupMemberBuff.data(), groupMembersBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tgroupMembers updated\n");
}
