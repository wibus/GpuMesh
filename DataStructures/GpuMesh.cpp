#include "GpuMesh.h"

#include <CellarWorkbench/Misc/Log.h>
#include <CellarWorkbench/GL/GlProgram.h>

#include "NodeGroups.h"

#include "Boundaries/Constraints/AbstractConstraint.h"

using namespace std;
using namespace cellar;


// CUDA Drivers Interface
void fetchCudaVerts(GpuVert* vertsBuff, size_t vertsLength);
void updateCudaVerts(const GpuVert* vertsBuff, size_t vertsLength);
void updateCudaTets(const std::vector<GpuTet>& tetsBuff);
void updateCudaPris(const std::vector<GpuPri>& prisBuff);
void updateCudaHexs(const std::vector<GpuHex>& hexsBuff);
void updateCudaTopo(
        const std::vector<GpuTopo>& toposBuff,
        const std::vector<GpuNeigVert>& neigVertsBuff,
        const std::vector<GpuNeigElem>& neigElemsBuff);
void updateCudaGroupMembers(
        const std::vector<GLuint>& groupMembersBuff);


GpuMesh::GpuMesh() :
    _vertSsbo(0),
    _topoSsbo(0),
    _priSsbo(0),
    _hexSsbo(0),
    _tetSsbo(0),
    _neigVertSsbo(0),
    _neigElemSsbo(0),
    _groupMembersSsbo(0)
{

}

GpuMesh::~GpuMesh()
{
    glDeleteBuffers(1, &_vertSsbo);
    glDeleteBuffers(1, &_tetSsbo);
    glDeleteBuffers(1, &_priSsbo);
    glDeleteBuffers(1, &_hexSsbo);
    glDeleteBuffers(1, &_topoSsbo);
    glDeleteBuffers(1, &_neigVertSsbo);
    glDeleteBuffers(1, &_neigElemSsbo);
    glDeleteBuffers(1, &_groupMembersSsbo);
}

void GpuMesh::clear()
{
    Mesh::clear();

    glDeleteBuffers(1, &_vertSsbo);
    glDeleteBuffers(1, &_tetSsbo);
    glDeleteBuffers(1, &_priSsbo);
    glDeleteBuffers(1, &_hexSsbo);
    glDeleteBuffers(1, &_topoSsbo);
    glDeleteBuffers(1, &_neigVertSsbo);
    glDeleteBuffers(1, &_neigElemSsbo);
    glDeleteBuffers(1, &_groupMembersSsbo);

    _vertSsbo = 0;
    _tetSsbo = 0;
    _priSsbo = 0;
    _hexSsbo = 0;
    _topoSsbo = 0;
    _neigVertSsbo = 0;
    _neigElemSsbo = 0;
    _groupMembersSsbo = 0;
}

void GpuMesh::compileTopology(bool updateGpu)
{
    Mesh::compileTopology(updateGpu);

    if(updateGpu)
    {
        if(_vertSsbo == 0)
        {
            getLog().postMessage(new Message('I', false,
                "Generating mesh shader storage buffers", "GpuMesh"));

            glGenBuffers(1, &_vertSsbo);
            glGenBuffers(1, &_tetSsbo);
            glGenBuffers(1, &_priSsbo);
            glGenBuffers(1, &_hexSsbo);
            glGenBuffers(1, &_topoSsbo);
            glGenBuffers(1, &_neigVertSsbo);
            glGenBuffers(1, &_neigElemSsbo);
            glGenBuffers(1, &_groupMembersSsbo);


            // Allocation GPU side vertex positions storage space
            size_t vertBuffSize = sizeof(GpuVert) * verts.size();
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
            glBufferData(GL_SHADER_STORAGE_BUFFER, vertBuffSize, nullptr, GL_DYNAMIC_COPY);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

            updateVerticesFromCpu();
        }

        updateGpuTopology();
    }
}


void GpuMesh::updateGpuTopology()
{
    // Send individual elements
    size_t tetCount = tets.size();
    std::vector<GpuTet> tetBuff(tetCount);
    for(int i=0; i < tetCount; ++i)
        tetBuff[i] = tets[i];

    updateCudaTets(tetBuff);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _tetSsbo);
    size_t tetSize = sizeof(decltype(tetBuff.front())) * tetBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, tetSize, tetBuff.data(), GL_STATIC_COPY);
    tetBuff.clear();
    tetBuff.shrink_to_fit();


    size_t priCount = pris.size();
    std::vector<GpuPri> priBuff(priCount);
    for(int i=0; i < priCount; ++i)
        priBuff[i] = pris[i];

    updateCudaPris(priBuff);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _priSsbo);
    size_t priSize = sizeof(decltype(priBuff.front())) * priBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, priSize, priBuff.data(), GL_STATIC_COPY);
    priBuff.clear();
    priBuff.shrink_to_fit();


    size_t hexCount = hexs.size();
    std::vector<GpuHex> hexBuff(hexCount);
    for(int i=0; i < hexCount; ++i)
        hexBuff[i] = hexs[i];

    updateCudaHexs(hexBuff);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _hexSsbo);
    size_t hexSize = sizeof(decltype(hexBuff.front())) * hexBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, hexSize, hexBuff.data(), GL_STATIC_COPY);
    hexBuff.clear();
    hexBuff.shrink_to_fit();


    // Topo descriptors
    size_t vertCount = verts.size();
    std::vector<GpuTopo> topoBuff(vertCount);
    std::vector<GpuNeigVert> neigVertBuff;
    std::vector<GpuNeigElem> neigElemBuff;

    int neigVertBase = 0;
    int neigElemBase = 0;
    for(int i=0; i < vertCount; ++i)
    {
        const MeshTopo& meshTopo = topos[i];
        uint neigVertCount = (uint)meshTopo.neighborVerts.size();
        uint neigElemCount = (uint)meshTopo.neighborElems.size();
        int type = meshTopo.snapToBoundary->isFixed() ? -1 :
                meshTopo.snapToBoundary->id();

        topoBuff[i] = GpuTopo(type,
            neigVertBase, neigVertCount,
            neigElemBase, neigElemCount);

		for (uint n = 0; n < neigVertCount; ++n)
            neigVertBuff.push_back(GpuNeigVert(meshTopo.neighborVerts[n]));

		for (uint n = 0; n < neigElemCount; ++n)
            neigElemBuff.push_back(GpuNeigElem(meshTopo.neighborElems[n]));

        neigVertBase += neigVertCount;
        neigElemBase += neigElemCount;
    }

    updateCudaTopo(topoBuff, neigVertBuff, neigElemBuff);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _topoSsbo);
    size_t topoSize = sizeof(decltype(topoBuff.front())) * topoBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, topoSize, topoBuff.data(), GL_STATIC_COPY);
    topoBuff.clear();
    topoBuff.shrink_to_fit();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _neigVertSsbo);
    size_t neigVertSize = sizeof(decltype(neigVertBuff.front())) * neigVertBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, neigVertSize, neigVertBuff.data(), GL_STATIC_COPY);
    neigVertBuff.clear();
    neigVertBuff.shrink_to_fit();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _neigElemSsbo);
    size_t neigElemSize = sizeof(decltype(neigElemBuff.front())) * neigElemBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, neigElemSize, neigElemBuff.data(), GL_STATIC_COPY);
    neigElemBuff.clear();
    neigElemBuff.shrink_to_fit();


    // Independent group members
    std::vector<GLuint> groupMemberBuff(
        nodeGroups().gpuGroupsBuffer().begin(),
        nodeGroups().gpuGroupsBuffer().end());

    updateCudaGroupMembers(groupMemberBuff);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _groupMembersSsbo);
    size_t membersSize = sizeof(decltype(groupMemberBuff.front())) * groupMemberBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, membersSize, groupMemberBuff.data(), GL_STATIC_COPY);    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    groupMemberBuff.clear();
    groupMemberBuff.shrink_to_fit();
}

void GpuMesh::updateVerticesFromCpu()
{
    size_t vertCount = verts.size();
    size_t vertBuffSize = sizeof(GpuVert) * vertCount;

    // Grow GLSL buffer if we've added some vertices
    GLint buffSize = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &buffSize);
    GLint cpuBuffSize = sizeof(decltype(verts.front())) * verts.size();

    if(buffSize != cpuBuffSize)
        glBufferData(GL_SHADER_STORAGE_BUFFER, cpuBuffSize, nullptr, GL_DYNAMIC_COPY);


    GpuVert* glslVerts = (GpuVert*) glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, vertBuffSize,
        GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

    for(int i=0; i < vertCount; ++i)
        glslVerts[i] = GpuVert(verts[i]);
    updateCudaVerts(glslVerts, verts.size());

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GpuMesh::updateVerticesFromGlsl()
{
    size_t vertCount = verts.size();
    size_t vertBuffSize = sizeof(GpuVert) * vertCount;

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    GpuVert* glslVerts = (GpuVert*) glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, vertBuffSize,
        GL_MAP_READ_BIT);

    updateCudaVerts(glslVerts, verts.size());
    for(int i=0; i < vertCount; ++i)
        verts[i].p = glm::dvec3(glslVerts[i].p);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GpuMesh::updateVerticesFromCuda()
{
    size_t vertCount = verts.size();
    size_t vertBuffSize = sizeof(GpuVert) * vertCount;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    GpuVert* glslVerts = (GpuVert*) glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, vertBuffSize,
        GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

    fetchCudaVerts(glslVerts, verts.size());
    for(int i=0; i < vertCount; ++i)
        verts[i].p = glm::dvec3(glslVerts[i].p);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

unsigned int GpuMesh::glBuffer(const EMeshBuffer& buffer) const
{
    switch(buffer)
    {
    case EMeshBuffer::VERT:             return _vertSsbo;
    case EMeshBuffer::TET:              return _tetSsbo;
    case EMeshBuffer::PRI:              return _priSsbo;
    case EMeshBuffer::HEX:              return _hexSsbo;
    case EMeshBuffer::TOPO:             return _topoSsbo;
    case EMeshBuffer::NEIG_VERT:        return _neigVertSsbo;
    case EMeshBuffer::NEIG_ELEM:        return _neigElemSsbo;
    case EMeshBuffer::GROUP_MEMBERS:    return _groupMembersSsbo;
    default : return 0;
    }
}

std::string GpuMesh::meshGeometryShaderName() const
{
    return ":/glsl/compute/Mesh.glsl";
}

unsigned int GpuMesh::bufferBinding(EBufferBinding binding) const
{
    switch(binding)
    {
    case EBufferBinding::EVALUATE_QUAL_BUFFER_BINDING :     return 8;
    case EBufferBinding::EVALUATE_HIST_BUFFER_BINDING :     return 9;
    case EBufferBinding::VERTEX_ACCUMS_BUFFER_BINDING :     return 10;
    case EBufferBinding::REF_VERTS_BUFFER_BINDING:          return 11;
    case EBufferBinding::REF_METRICS_BUFFER_BINDING:        return 12;
    case EBufferBinding::KD_TETS_BUFFER_BINDING :           return 13;
    case EBufferBinding::KD_NODES_BUFFER_BINDING :          return 14;
    case EBufferBinding::LOCAL_TETS_BUFFER_BINDING :        return 15;
    }

    return 0;
}

void GpuMesh::bindShaderStorageBuffers() const
{
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _vertSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _tetSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _priSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _hexSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _topoSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _neigVertSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _neigElemSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _groupMembersSsbo);
}
