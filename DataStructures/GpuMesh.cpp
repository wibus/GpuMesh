#include "GpuMesh.h"

#include <iostream>

using namespace std;


GpuMesh::GpuMesh() :
    _vertSsbo(0),
    _topoSsbo(0),
    _neigSsbo(0),
    _qualSsbo(0),
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
    glDeleteBuffers(1, &_qualSsbo);
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
    glDeleteBuffers(1, &_qualSsbo);
    glDeleteBuffers(1, &_tetSsbo);
    glDeleteBuffers(1, &_priSsbo);
    glDeleteBuffers(1, &_hexSsbo);

    _vertSsbo = 0;
    _topoSsbo = 0;
    _neigSsbo = 0;
    _qualSsbo = 0;
    _tetSsbo = 0;
    _priSsbo = 0;
    _hexSsbo = 0;
}

void GpuMesh::compileTopoly()
{
    Mesh::compileTopoly();

    cout << "Generating mesh shader storage buffers" << endl;

    if(_vertSsbo == 0)
    {
        glGenBuffers(1, &_vertSsbo);
        glGenBuffers(1, &_topoSsbo);
        glGenBuffers(1, &_neigSsbo);
        glGenBuffers(1, &_qualSsbo);
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
    std::vector<int> neigBuff;

    int base = 0;
    for(int i=0; i < nbVert; ++i)
    {
        const MeshTopo& meshTopo = topo[i];
        int count = meshTopo.neighbors.size();
        int type = meshTopo.isFixed ? -1 :
                meshTopo.boundaryCallback.id();

        topoBuff[i] = GpuTopo(type, base, count);

        for(int n=0; n < count; ++n)
            neigBuff.push_back(meshTopo.neighbors[n]);

        base += count;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _topoSsbo);
    size_t topoSize = sizeof(decltype(topoBuff.front())) * topoBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, topoSize, topoBuff.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _neigSsbo);
    size_t neigSize = sizeof(decltype(neigBuff.front())) * neigBuff.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, neigSize, neigBuff.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _tetSsbo);
    size_t tetSize = sizeof(decltype(tetra.front())) * tetra.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, tetSize, tetra.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _priSsbo);
    size_t priSize = sizeof(decltype(prism.front())) * prism.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, priSize, prism.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _hexSsbo);
    size_t hexSize = sizeof(decltype(hexa.front())) * hexa.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, hexSize, hexa.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GpuMesh::updateGpuVertices()
{
    int nbVert = vertCount();
    std::vector<glm::vec4> buff(nbVert);
    size_t size = sizeof(decltype(buff.front())) * nbVert;

    for(int i=0; i < nbVert; ++i)
        buff[i] = glm::vec4(vert[i].p, 0.0);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, buff.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GpuMesh::updateCpuVertices()
{
    int nbVert = vertCount();
    std::vector<glm::vec4> buff(nbVert);
    size_t size = sizeof(decltype(buff.front())) * nbVert;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size, buff.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    for(int i=0; i < nbVert; ++i)
        vert[i].p = glm::dvec3(buff[i]);
}

unsigned int GpuMesh::glBuffer(const EMeshBuffer& buffer) const
{
    switch(buffer)
    {
    case EMeshBuffer::VERT: return _vertSsbo;
    case EMeshBuffer::TOPO: return _topoSsbo;
    case EMeshBuffer::NEIG: return _neigSsbo;
    case EMeshBuffer::QUAL: return _qualSsbo;
    case EMeshBuffer::TET:  return _tetSsbo;
    case EMeshBuffer::PRI:  return _priSsbo;
    case EMeshBuffer::HEX:  return _hexSsbo;
    default : return 0;
    }
}
