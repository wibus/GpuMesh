#include "Mesh.cuh"

#include "DataStructures/GpuMesh.h"


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

void fetchCudaVerts(GpuVert* vertsBuff, size_t vertsLength)
{
    size_t vertsBuffSize = sizeof(GpuVert) *vertsLength;
    cudaMemcpy(vertsBuff, d_verts, vertsBuffSize, cudaMemcpyDeviceToHost);
}

void updateCudaVerts(const GpuVert* vertsBuff, size_t vertsLength)
{
    size_t vertsBuffSize = sizeof(GpuVert) * vertsLength;
    if(d_verts == nullptr || d_vertsLength != vertsLength)
    {
        cudaFree(d_verts);
        if(!vertsLength) d_verts = nullptr;
        else cudaMalloc(&d_verts, vertsBuffSize);
        cudaMemcpyToSymbol(verts, &d_verts, sizeof(d_verts));

        d_vertsLength = vertsLength;
        cudaMemcpyToSymbol(verts_length, &vertsLength, sizeof(uint));
    }

    cudaMemcpy(d_verts, vertsBuff, vertsBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tverts updated\n");
}


size_t d_tetsLength = 0;
GpuTet* d_tets = nullptr;
void updateCudaTets(const std::vector<GpuTet>& tetsBuff)
{
    uint tetsLength = tetsBuff.size();
    size_t tetsBuffSize = sizeof(decltype(tetsBuff.front())) * tetsLength;
    if(d_tets == nullptr || d_tetsLength != tetsLength)
    {
        cudaFree(d_tets);
        if(!tetsLength) d_tets = nullptr;
        else cudaMalloc(&d_tets, tetsBuffSize);
        cudaMemcpyToSymbol(tets, &d_tets, sizeof(d_tets));

        d_tetsLength = tetsLength;
        cudaMemcpyToSymbol(tets_length, &tetsLength, sizeof(uint));
    }

    cudaMemcpy(d_tets, tetsBuff.data(), tetsBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \ttets updated\n");
}


size_t d_prisLength = 0;
GpuPri* d_pris = nullptr;
void updateCudaPris(const std::vector<GpuPri>& prisBuff)
{
    uint prisLength = prisBuff.size();
    size_t prisBuffSize = sizeof(decltype(prisBuff.front())) * prisLength;
    if(d_pris == nullptr || d_prisLength != prisLength)
    {
        cudaFree(d_pris);
        if(!prisLength) d_pris = nullptr;
        else cudaMalloc(&d_pris, prisBuffSize);
        cudaMemcpyToSymbol(pris, &d_pris, sizeof(d_pris));

        d_prisLength = prisLength;
        cudaMemcpyToSymbol(pris_length, &prisLength, sizeof(uint));
    }

    cudaMemcpy(d_pris, prisBuff.data(), prisBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tpris updated\n");
}


size_t d_hexsLength = 0;
GpuHex* d_hexs = nullptr;
void updateCudaHexs(const std::vector<GpuHex>& hexsBuff)
{
    uint hexsLength = hexsBuff.size();
    size_t hexsBuffSize = sizeof(decltype(hexsBuff.front())) * hexsLength;
    if(d_hexs == nullptr || d_hexsLength != hexsLength)
    {
        cudaFree(d_hexs);
        if(!hexsLength) d_hexs = nullptr;
        else cudaMalloc(&d_hexs, hexsBuffSize);
        cudaMemcpyToSymbol(hexs, &d_hexs, sizeof(d_hexs));

        d_hexsLength = hexsLength;
        cudaMemcpyToSymbol(hexs_length, &hexsLength, sizeof(uint));
    }

    cudaMemcpy(d_hexs, hexsBuff.data(), hexsBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \thexs updated\n");
}


size_t d_toposLength = 0;
Topo* d_topos = nullptr;

size_t d_neigVertsLength = 0;
NeigVert* d_neigVerts = nullptr;

size_t d_neigElemsLength = 0;
NeigElem* d_neigElems = nullptr;

void updateCudaTopo(
        const std::vector<GpuTopo>& toposBuff,
        const std::vector<GpuNeigVert>& neigVertsBuff,
        const std::vector<GpuNeigElem>& neigElemsBuff)
{
    // Topologies
    uint toposLength = toposBuff.size();
    size_t toposBuffSize = sizeof(decltype(toposBuff.front())) * toposLength;
    if(d_topos == nullptr || d_toposLength != toposLength)
    {
        cudaFree(d_topos);
        if(!toposLength) d_topos = nullptr;
        else cudaMalloc(&d_topos, toposBuffSize);
        cudaMemcpyToSymbol(topos, &d_topos, sizeof(d_topos));

        d_toposLength = toposLength;
        cudaMemcpyToSymbol(topos_length, &toposLength, sizeof(uint));
    }

    cudaMemcpy(d_topos, toposBuff.data(), toposBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \ttopos updated\n");


    // Neighbor vertices
    uint neigVertsLength = neigVertsBuff.size();
    size_t neigVertsBuffSize = sizeof(decltype(neigVertsBuff.front())) * neigVertsLength;
    if(d_neigVerts == nullptr || d_neigVertsLength != neigVertsLength)
    {
        cudaFree(d_neigVerts);
        if(!neigVertsLength) d_neigVerts = nullptr;
        else cudaMalloc(&d_neigVerts, neigVertsBuffSize);
        cudaMemcpyToSymbol(neigVerts, &d_neigVerts, sizeof(d_neigVerts));

        d_neigVertsLength = neigVertsLength;
        cudaMemcpyToSymbol(neigVerts_length, &neigVertsLength, sizeof(uint));
    }

    cudaMemcpy(d_neigVerts, neigVertsBuff.data(), neigVertsBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tneigVerts updated\n");


    // Neighbor elements
    uint neigElemsLength = neigElemsBuff.size();
    size_t neigElemsBuffSize = sizeof(decltype(neigElemsBuff.front())) * neigElemsLength;
    if(d_neigElems == nullptr || d_neigElemsLength != neigElemsLength)
    {
        cudaFree(d_neigElems);
        if(!neigElemsLength) d_neigElems = nullptr;
        else cudaMalloc(&d_neigElems, neigElemsBuffSize);
        cudaMemcpyToSymbol(neigElems, &d_neigElems, sizeof(d_neigElems));

        d_neigElemsLength = neigElemsLength;
        cudaMemcpyToSymbol(neigElems_length, &neigElemsLength, sizeof(uint));
    }

    cudaMemcpy(d_neigElems, neigElemsBuff.data(), neigElemsBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tneigElems updated\n");
}


size_t d_groupMembersLength = 0;
uint* d_groupMembers = nullptr;
void updateCudaGroupMembers(const std::vector<GLuint>& groupMembersBuff)
{
    // Group members
    uint groupMembersLength = groupMembersBuff.size();
    size_t groupMembersBuffSize = sizeof(decltype(groupMembersBuff.front())) * groupMembersLength;
    if(d_groupMembers == nullptr || d_groupMembersLength != groupMembersLength)
    {
        cudaFree(d_groupMembers);
        if(!groupMembersLength) d_groupMembers = nullptr;
        else cudaMalloc(&d_groupMembers, groupMembersBuffSize);
        cudaMemcpyToSymbol(groupMembers, &d_groupMembers, sizeof(d_groupMembers));

        d_groupMembersLength = groupMembersLength;
        cudaMemcpyToSymbol(groupMembers_length, &groupMembersLength, sizeof(uint));
    }

    cudaMemcpy(d_groupMembers, groupMembersBuff.data(), groupMembersBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tgroupMembers updated\n");
}
