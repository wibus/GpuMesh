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

__constant__ Tri MeshTet_tris[4] = {
    {{1, 2, 3}},
    {{0, 3, 2}},
    {{0, 1, 3}},
    {{0, 2, 1}}};


// Reference mesh
__constant__ uint refVerts_length;
__device__ Vert* refVerts;

__constant__ uint refMetrics_length;
__device__ mat4* refMetrics;


// Debug
bool verboseCuda = true;


// CUDA Drivers
size_t d_vertsLength = 0;
GpuVert* d_verts = nullptr;

void fetchCudaVerts(std::vector<GpuVert>& vertsBuff)
{
    if(vertsBuff.size() != d_vertsLength)
    {
        printf("I -> CUDA \tverts CPU and GPU buffer sizes mismatch\n");
        assert(vertsBuff.size() == d_vertsLength);
    }

    uint vertsLength = vertsBuff.size();
    size_t vertsBuffSize = sizeof(GpuVert) * vertsLength;
    cudaMemcpy(vertsBuff.data(), d_verts, vertsBuffSize, cudaMemcpyDeviceToHost);

    if(verboseCuda)
        printf("I -> CUDA \tverts fetched\n");

    cudaCheckErrors("Verts fetch");
}

void updateCudaVerts(const std::vector<GpuVert>& vertsBuff)
{
    uint vertsLength = vertsBuff.size();
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

    cudaMemcpy(d_verts, vertsBuff.data(), vertsBuffSize, cudaMemcpyHostToDevice);

    if(verboseCuda)
        printf("I -> CUDA \tverts updated\n");

    cudaCheckErrors("Verts update");
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

    if(verboseCuda)
        printf("I -> CUDA \ttets updated\n");

    cudaCheckErrors("Tets update");
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

    if(verboseCuda)
        printf("I -> CUDA \tpris updated\n");

    cudaCheckErrors("Pris update");
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

    if(verboseCuda)
        printf("I -> CUDA \thexs updated\n");

    cudaCheckErrors("Hexs update");
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

    if(verboseCuda)
        printf("I -> CUDA \ttopos updated\n");

    cudaCheckErrors("Topos update");


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

    if(verboseCuda)
        printf("I -> CUDA \tneigVerts updated\n");

    cudaCheckErrors("Neig verts update");


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

    if(verboseCuda)
        printf("I -> CUDA \tneigElems updated\n");

    cudaCheckErrors("Neig elems update");
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

    if(verboseCuda)
        printf("I -> CUDA \tgroupMembers updated\n");

    cudaCheckErrors("Group members update");
}


size_t d_refVertsLength = 0;
Vert* d_refVerts = nullptr;
void updateCudaRefVerts(
        const std::vector<GpuVert>& refVertsBuff)
{
    // Reference mesh vertices
    uint refVertsLength = refVertsBuff.size();
    size_t refVertsBuffSize = sizeof(decltype(refVertsBuff.front())) * refVertsLength;
    if(d_refVerts == nullptr || d_refVertsLength != refVertsLength)
    {
        cudaFree(d_refVerts);
        if(!refVertsLength) d_refVerts = nullptr;
        else cudaMalloc(&d_refVerts, refVertsBuffSize);
        cudaMemcpyToSymbol(refVerts, &d_refVerts, sizeof(d_refVerts));

        d_refVertsLength = refVertsLength;
        cudaMemcpyToSymbol(refVerts_length, &refVertsLength, sizeof(uint));
    }

    cudaMemcpy(d_refVerts, refVertsBuff.data(), refVertsBuffSize, cudaMemcpyHostToDevice);

    if(verboseCuda)
        printf("I -> CUDA \tRef Vertices updated\n");

    cudaCheckErrors("Ref verts update");
}


size_t d_refMetricsLength = 0;
mat4* d_refMetrics = nullptr;
void updateCudaRefMetrics(
        const std::vector<glm::mat4>& refMetricsBuff)
{
    // Reference mesh metrics
    uint refMetricsLength = refMetricsBuff.size();
    size_t refMetricsBuffSize = sizeof(decltype(refMetricsBuff.front())) * refMetricsLength;
    if(d_refMetrics == nullptr || d_refMetricsLength != refMetricsLength)
    {
        cudaFree(d_refMetrics);
        if(!refMetricsLength) d_refMetrics = nullptr;
        else cudaMalloc(&d_refMetrics, refMetricsBuffSize);
        cudaMemcpyToSymbol(refMetrics, &d_refMetrics, sizeof(d_refMetrics));

        d_refMetricsLength = refMetricsLength;
        cudaMemcpyToSymbol(refMetrics_length, &refMetricsLength, sizeof(uint));
    }

    cudaMemcpy(d_refMetrics, refMetricsBuff.data(), refMetricsBuffSize, cudaMemcpyHostToDevice);

    if(verboseCuda)
        printf("I -> CUDA \tRef Metrics updated\n");

    cudaCheckErrors("Ref metrics update");
}

void cppCudaCheckErrors(const char* msg)
{
    cudaCheckErrors(msg);
}
