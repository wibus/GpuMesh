#include "GpuMesh.h"

#include <CellarWorkbench/Misc/Log.h>
#include <CellarWorkbench/GL/GlProgram.h>

using namespace std;
using namespace cellar;


GpuMesh::GpuMesh() :
    _vertSsbo(0),
    _topoSsbo(0),
    _priSsbo(0),
    _hexSsbo(0),
    _tetSsbo(0),
    _neigVertSsbo(0),
    _neigElemSsbo(0)
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

    _vertSsbo = 0;
    _tetSsbo = 0;
    _priSsbo = 0;
    _hexSsbo = 0;
    _topoSsbo = 0;
    _neigVertSsbo = 0;
    _neigElemSsbo = 0;
}

void GpuMesh::compileTopoly()
{
    Mesh::compileTopoly();

    getLog().postMessage(new Message('I', false,
        "Generating mesh shader storage buffers", "GpuMesh"));

    if(_vertSsbo == 0)
    {
        glGenBuffers(1, &_vertSsbo);
        glGenBuffers(1, &_tetSsbo);
        glGenBuffers(1, &_priSsbo);
        glGenBuffers(1, &_hexSsbo);
        glGenBuffers(1, &_topoSsbo);
        glGenBuffers(1, &_neigVertSsbo);
        glGenBuffers(1, &_neigElemSsbo);
    }

    updateGpuVertices();
    updateGpuTopoly();
}

void GpuMesh::updateGpuTopoly()
{
    // Send mesh topology
    int nbVert = vertCount();
    std::vector<GpuTopo> topoBuff(nbVert);
    std::vector<GpuNeigVert> neigVertBuff;
    std::vector<GpuNeigElem> neigElemBuff;

    int neigVertBase = 0;
    int neigElemBase = 0;
    for(int i=0; i < nbVert; ++i)
    {
        const MeshTopo& meshTopo = topo[i];
        int neigVertCount = meshTopo.neighborVerts.size();
        int neigElemCount = meshTopo.neighborElems.size();
        int type = meshTopo.isFixed ? -1 :
                meshTopo.snapToBoundary->id();

        topoBuff[i] = GpuTopo(type, neigVertBase, neigVertCount,
                                    neigElemBase, neigElemCount);

        for(int n=0; n < neigVertCount; ++n)
            neigVertBuff.push_back(GpuNeigVert(meshTopo.neighborVerts[n]));

        for(int n=0; n < neigElemCount; ++n)
            neigElemBuff.push_back(GpuNeigElem(meshTopo.neighborElems[n]));

        neigVertBase += neigVertCount;
        neigElemBase += neigElemCount;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _topoSsbo);
    size_t topoSize = sizeof(decltype(topoBuff.front())) * topoBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, topoSize, topoBuff.data(), GL_STATIC_DRAW);
    topoBuff.clear();
    topoBuff.shrink_to_fit();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _neigVertSsbo);
    size_t neigVertSize = sizeof(decltype(neigVertBuff.front())) * neigVertBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, neigVertSize, neigVertBuff.data(), GL_STATIC_DRAW);
    neigVertBuff.clear();
    neigVertBuff.shrink_to_fit();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _neigElemSsbo);
    size_t neigElemSize = sizeof(decltype(neigElemBuff.front())) * neigElemBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, neigElemSize, neigElemBuff.data(), GL_STATIC_DRAW);
    neigElemBuff.clear();
    neigElemBuff.shrink_to_fit();


    // Send individual elements
    size_t tetCount = tetra.size();
    std::vector<GpuTet> tetBuff(tetCount);
    for(int i=0; i < tetCount; ++i)
        tetBuff[i] = tetra[i];

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _tetSsbo);
    size_t tetSize = sizeof(decltype(tetBuff.front())) * tetBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, tetSize, tetBuff.data(), GL_STATIC_DRAW);
    tetBuff.clear();
    tetBuff.shrink_to_fit();


    size_t priCount = prism.size();
    std::vector<GpuPri> priBuff(priCount);
    for(int i=0; i < priCount; ++i)
        priBuff[i] = prism[i];

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _priSsbo);
    size_t priSize = sizeof(decltype(priBuff.front())) * priBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, priSize, priBuff.data(), GL_STATIC_DRAW);
    priBuff.clear();
    priBuff.shrink_to_fit();


    size_t hexCount = hexa.size();
    std::vector<GpuHex> hexBuff(hexCount);
    for(int i=0; i < hexCount; ++i)
        hexBuff[i] = hexa[i];

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _hexSsbo);
    size_t hexSize = sizeof(decltype(hexBuff.front())) * hexBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, hexSize, hexBuff.data(), GL_STATIC_DRAW);
    hexBuff.clear();
    hexBuff.shrink_to_fit();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GpuMesh::updateGpuVertices()
{
    int nbVert = vertCount();
    std::vector<GpuVert> buff(nbVert);
    size_t size = sizeof(decltype(buff.front())) * nbVert;

    for(int i=0; i < nbVert; ++i)
        buff[i] = GpuVert(vert[i]);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, buff.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GpuMesh::updateCpuVertices()
{
    int nbVert = vertCount();
    std::vector<GpuVert> buff(nbVert);
    size_t size = sizeof(decltype(buff.front())) * nbVert;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size, buff.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    for(int i=0; i < nbVert; ++i)
        vert[i] = buff[i];
}

unsigned int GpuMesh::glBuffer(const EMeshBuffer& buffer) const
{
    switch(buffer)
    {
    case EMeshBuffer::VERT: return _vertSsbo;
    case EMeshBuffer::TET:  return _tetSsbo;
    case EMeshBuffer::PRI:  return _priSsbo;
    case EMeshBuffer::HEX:  return _hexSsbo;
    case EMeshBuffer::TOPO: return _topoSsbo;
    case EMeshBuffer::NEIG_VERT: return _neigVertSsbo;
    case EMeshBuffer::NEIG_ELEM: return _neigElemSsbo;
    default : return 0;
    }
}

std::string GpuMesh::meshGeometryShaderName() const
{
    return ":/shaders/compute/Mesh.glsl";
}

void GpuMesh::uploadGeometry(cellar::GlProgram& program) const
{
    program.pushProgram();
    uploadElement(program, "TET",
        MeshTet::EDGE_COUNT, MeshTet::edges,
        MeshTet::TRI_COUNT,  MeshTet::tris,
        MeshTet::TET_COUNT,  MeshTet::tets);

    uploadElement(program, "PRI",
        MeshPri::EDGE_COUNT, MeshPri::edges,
        MeshPri::TRI_COUNT,  MeshPri::tris,
        MeshPri::TET_COUNT,  MeshPri::tets);

    uploadElement(program, "HEX",
        MeshHex::EDGE_COUNT, MeshHex::edges,
        MeshHex::TRI_COUNT,  MeshHex::tris,
        MeshHex::TET_COUNT,  MeshHex::tets);
    program.popProgram();
}

void GpuMesh::uploadElement(
        cellar::GlProgram& program,
        const std::string& prefix,
        int edgeCount, const MeshEdge edges[],
        int triCount,  const MeshTri tris[],
        int tetCount,  const MeshTet tets[]) const
{
    for(int i=0; i < edgeCount; ++i)
    {
        program.setUnsigned(prefix + "_EDGES[" + to_string(i) + "].v[0]", edges[i][0]);
        program.setUnsigned(prefix + "_EDGES[" + to_string(i) + "].v[1]", edges[i][1]);
    }

    for(int i=0; i < triCount; ++i)
    {
        program.setUnsigned(prefix + "_TRIS[" + to_string(i) + "].v[0]", tris[i][0]);
        program.setUnsigned(prefix + "_TRIS[" + to_string(i) + "].v[1]", tris[i][1]);
        program.setUnsigned(prefix + "_TRIS[" + to_string(i) + "].v[2]", tris[i][2]);
    }

    for(int i=0; i < tetCount; ++i)
    {
        program.setUnsigned(prefix + "_TETS[" + to_string(i) + "].v[0]", tets[i][0]);
        program.setUnsigned(prefix + "_TETS[" + to_string(i) + "].v[1]", tets[i][1]);
        program.setUnsigned(prefix + "_TETS[" + to_string(i) + "].v[2]", tets[i][2]);
        program.setUnsigned(prefix + "_TETS[" + to_string(i) + "].v[3]", tets[i][3]);
    }
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
}

size_t GpuMesh::firstFreeBufferBinding() const
{
    return 7;
}
