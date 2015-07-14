#include "GpuMesh.h"

#include <CellarWorkbench/Misc/Log.h>
#include <CellarWorkbench/GL/GlProgram.h>

using namespace std;
using namespace cellar;


GpuMesh::GpuMesh() :
    _vertSsbo(0),
    _topoSsbo(0),
    _neigSsbo(0),
    _tetSsbo(0),
    _priSsbo(0),
    _hexSsbo(0)
{

}

GpuMesh::~GpuMesh()
{
    glDeleteBuffers(1, &_vertSsbo);
    glDeleteBuffers(1, &_topoSsbo);
    glDeleteBuffers(1, &_neigSsbo);
    glDeleteBuffers(1, &_tetSsbo);
    glDeleteBuffers(1, &_priSsbo);
    glDeleteBuffers(1, &_hexSsbo);
}

void GpuMesh::clear()
{
    Mesh::clear();

    glDeleteBuffers(1, &_vertSsbo);
    glDeleteBuffers(1, &_topoSsbo);
    glDeleteBuffers(1, &_neigSsbo);
    glDeleteBuffers(1, &_tetSsbo);
    glDeleteBuffers(1, &_priSsbo);
    glDeleteBuffers(1, &_hexSsbo);

    _vertSsbo = 0;
    _topoSsbo = 0;
    _neigSsbo = 0;
    _tetSsbo = 0;
    _priSsbo = 0;
    _hexSsbo = 0;
}

void GpuMesh::compileTopoly()
{
    Mesh::compileTopoly();

    getLog().postMessage(new Message('I', false,
        "Generating mesh shader storage buffers", "GpuMesh"));

    if(_vertSsbo == 0)
    {
        glGenBuffers(1, &_vertSsbo);
        glGenBuffers(1, &_topoSsbo);
        glGenBuffers(1, &_neigSsbo);
        glGenBuffers(1, &_tetSsbo);
        glGenBuffers(1, &_priSsbo);
        glGenBuffers(1, &_hexSsbo);
    }

    updateGpuVertices();
    updateGpuTopoly();
}

void GpuMesh::updateGpuTopoly()
{
    int nbVert = vertCount();
    std::vector<GpuTopo> topoBuff(nbVert);
    std::vector<GpuNeig> neigBuff;

    int base = 0;
    for(int i=0; i < nbVert; ++i)
    {
        const MeshTopo& meshTopo = topo[i];
        int count = meshTopo.neighbors.size();
        int type = meshTopo.isFixed ? -1 :
                meshTopo.boundaryCallback.id();

        topoBuff[i] = GpuTopo(type, base, count);

        for(int n=0; n < count; ++n)
            neigBuff.push_back(GpuNeig(meshTopo.neighbors[n]));

        base += count;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _topoSsbo);
    size_t topoSize = sizeof(decltype(topoBuff.front())) * topoBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, topoSize, topoBuff.data(), GL_STATIC_DRAW);
    topoBuff.clear();
    topoBuff.shrink_to_fit();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _neigSsbo);
    size_t neigSize = sizeof(decltype(neigBuff.front())) * neigBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, neigSize, neigBuff.data(), GL_STATIC_DRAW);
    neigBuff.clear();
    neigBuff.shrink_to_fit();


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
    case EMeshBuffer::TOPO: return _topoSsbo;
    case EMeshBuffer::NEIG: return _neigSsbo;
    case EMeshBuffer::TET:  return _tetSsbo;
    case EMeshBuffer::PRI:  return _priSsbo;
    case EMeshBuffer::HEX:  return _hexSsbo;
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
        program.setInt(prefix + "_EDGES[" + to_string(i) + "].v[0]", edges[i][0]);
        program.setInt(prefix + "_EDGES[" + to_string(i) + "].v[1]", edges[i][1]);
    }

    for(int i=0; i < triCount; ++i)
    {
        program.setInt(prefix + "_TRIS[" + to_string(i) + "].v[0]", tris[i][0]);
        program.setInt(prefix + "_TRIS[" + to_string(i) + "].v[1]", tris[i][1]);
        program.setInt(prefix + "_TRIS[" + to_string(i) + "].v[2]", tris[i][2]);
    }

    for(int i=0; i < tetCount; ++i)
    {
        program.setInt(prefix + "_TETS[" + to_string(i) + "].v[0]", tets[i][0]);
        program.setInt(prefix + "_TETS[" + to_string(i) + "].v[1]", tets[i][1]);
        program.setInt(prefix + "_TETS[" + to_string(i) + "].v[2]", tets[i][2]);
        program.setInt(prefix + "_TETS[" + to_string(i) + "].v[3]", tets[i][3]);
    }
}

void GpuMesh::bindShaderStorageBuffers() const
{
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _vertSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _tetSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _priSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _hexSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _topoSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _neigSsbo);
}
