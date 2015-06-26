#include "GpuLaplacianSmoother.h"

#include <iostream>

using namespace std;


struct Topo
{
    // Type of vertex :
    //  * -1 = free
    //  *  0 = fixed
    //  * >0 = boundary
    int type;

    // Neighbors list start location
    int base;

    // Neighbors count
    int count;

    int pad;

    Topo() : type(0), base(0), count(0) {}
    Topo(int type, int base, int count) : type(type), base(base), count(count) {}
};

GpuLaplacianSmoother::GpuLaplacianSmoother(Mesh &mesh,
        double moveFactor,
        double gainThreshold) :
    AbstractSmoother(mesh, moveFactor, gainThreshold),
    _initialized(false),
    _topologyChanged(true),
    _vertSsbo(0),
    _topoSsbo(0),
    _neigSsbo(0)
{

}

GpuLaplacianSmoother::~GpuLaplacianSmoother()
{

}

void GpuLaplacianSmoother::smoothMesh()
{
    if(!_initialized)
    {
        initializeProgram();

        _initialized = true;
    }

    if(_topologyChanged)
    {
        updateTopology();

        _topologyChanged = false;
    }

    int vertCount = _mesh.vertCount();

    _smoothingProgram.pushProgram();
    _smoothingProgram.setInt("VertCount", vertCount);
    _smoothingProgram.setFloat("MoveCoeff", _moveFactor);

    evaluateInitialMeshQuality();

    while(evaluateIterationMeshQuality())
    {
        glDispatchCompute(ceil(vertCount / 256.0), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
        glm::vec4* vertPos = (glm::vec4*) glMapBuffer(
                GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        for(int i=0; i<vertCount; ++i)
        {
            _mesh.vert[i].p = glm::dvec3(vertPos[i]);
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }

    _smoothingProgram.popProgram();
}

void GpuLaplacianSmoother::initializeProgram()
{
    cout << "Initializing Laplacian smoothing compute shader" << endl;
    _smoothingProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/LaplacianSmoothing.glsl");
    _smoothingProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/ElbowPipeBoundaries.glsl");
    _smoothingProgram.link();


    glGenBuffers(1, &_vertSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _vertSsbo);

    glGenBuffers(1, &_topoSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _topoSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _topoSsbo);

    glGenBuffers(1, &_neigSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _neigSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _neigSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GpuLaplacianSmoother::updateTopology()
{
    cout << "Updating topology shader storage buffers" << endl;

    int vertCount = _mesh.vertCount();

    vector<glm::vec4> vert(vertCount);
    vector<Topo> topo(vertCount);
    vector<glm::ivec4> neig;

    int base = 0;
    for(int i=0; i < vertCount; ++i)
    {
        const MeshTopo& meshTopo = _mesh.topo[i];
        int type = meshTopo.isFixed ? -1 : meshTopo.boundaryCallback.id();
        int count = meshTopo.neighbors.size();

        vert[i] = glm::vec4(_mesh.vert[i].p, 0.0);
        topo[i] = Topo(type, base, count);

        for(int n=0; n < count; ++n)
            neig.push_back(glm::ivec4(meshTopo.neighbors[n]));

        base += count;
    }


    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _vertSsbo);
    size_t vertSize = sizeof(decltype(vert.front())) * vert.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, vertSize, vert.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _topoSsbo);
    size_t topoSize = sizeof(decltype(topo.front())) * topo.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, topoSize, topo.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _neigSsbo);
    size_t neigSize = sizeof(decltype(neig.front())) * neig.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, neigSize, neig.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}
