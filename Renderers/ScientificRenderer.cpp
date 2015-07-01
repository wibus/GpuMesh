#include "ScientificRenderer.h"

#include <iostream>

#include <GLM/gtc/matrix_transform.hpp>

#include <Scaena/StageManagement/Event/KeyboardEvent.h>
#include <Scaena/StageManagement/Event/SynchronousKeyboard.h>
#include <Scaena/StageManagement/Event/SynchronousMouse.h>

#include "Evaluators/GpuEvaluator.h"

using namespace std;
using namespace cellar;


ScientificRenderer::ScientificRenderer() :
    _vao(0),
    _vbo(0),
    _ibo(0),
    _lightMode(0),
    _lineWidth(5.0f),
    _pointRadius(0.002f),
    _isPhysicalCut(true),
    _evaluator(new GpuEvaluator())
{
}


ScientificRenderer::~ScientificRenderer()
{
    clearResources();
}

void ScientificRenderer::notify(cellar::CameraMsg& msg)
{
    if(msg.change == CameraMsg::EChange::VIEWPORT)
    {
        const glm::ivec2& viewport = msg.camera.viewport();

        // Camera projection
        const float n = 0.1;
        const float f = 12.0;
        glm::mat4 proj = glm::perspectiveFov(
                glm::pi<float>() / 6,
                (float) viewport.x,
                (float) viewport.y,
                n, f);

        _pointSphereProgram.pushProgram();
        _pointSphereProgram.setMat4f("ProjMat", proj);
        _pointSphereProgram.setVec2f("Viewport", viewport);
        _pointSphereProgram.popProgram();

        _wireframeProgram.pushProgram();
        _wireframeProgram.setMat4f("ProjMat", proj);
        _wireframeProgram.setVec2f("Viewport", viewport);
        _wireframeProgram.popProgram();
    }
}

void ScientificRenderer::updateCamera(const glm::mat4& view,
                                      const glm::vec3& pos)
{
    _pointSphereProgram.pushProgram();
    _pointSphereProgram.setMat4f("ViewMat", view);
    _pointSphereProgram.setVec3f("CameraPosition", pos);
    _pointSphereProgram.popProgram();

    _wireframeProgram.pushProgram();
    _wireframeProgram.setMat4f("ViewMat", view);
    _wireframeProgram.setVec3f("CameraPosition", pos);
    _wireframeProgram.popProgram();
}

void ScientificRenderer::updateLight(const glm::mat4&,
                                     const glm::vec3& pos)
{
    _pointSphereProgram.pushProgram();
    _pointSphereProgram.setVec3f("LightDirection", pos);
    _pointSphereProgram.popProgram();

    _wireframeProgram.pushProgram();
    _wireframeProgram.setVec3f("LightDirection", pos);
    _wireframeProgram.popProgram();
}

void ScientificRenderer::updateCutPlane(const glm::dvec4& cutEq)
{
    _cutPlane = cutEq;

    if(_isPhysicalCut)
    {
        _buffNeedUpdate = true;
    }

    _pointSphereProgram.pushProgram();
    _pointSphereProgram.setVec4f("CutPlaneEq", _cutPlane);
    _pointSphereProgram.popProgram();

    _wireframeProgram.pushProgram();
    _wireframeProgram.setVec4f("CutPlaneEq",_cutPlane);
    _wireframeProgram.popProgram();
}

void ScientificRenderer::handleKeyPress(const scaena::KeyboardEvent& event)
{
    if(event.getAscii() == 'X')
    {
        _lightMode = (_lightMode + 1) % 2;

        _pointSphereProgram.pushProgram();
        _pointSphereProgram.setInt("LightMode", _lightMode);
        _pointSphereProgram.popProgram();

        _wireframeProgram.pushProgram();
        _wireframeProgram.setInt("LightMode", _lightMode);
        _wireframeProgram.popProgram();

        const char* rep = (_lightMode ? "Phong" : "Diffuse") ;
        cout << "Using " << rep << " light mode" << endl;

    }
    else if(event.getAscii() == 'C')
    {
        _isPhysicalCut = !_isPhysicalCut;
        updateCutPlane(_cutPlane);
        _buffNeedUpdate = true;

        const char* rep = (_isPhysicalCut ? "true" : "false") ;
        cout << "Physical cut : " << rep << endl;
    }
}

void ScientificRenderer::handleInputs(const scaena::SynchronousKeyboard& keyboard,
                          const scaena::SynchronousMouse& mouse)
{

}

void ScientificRenderer::updateGeometry(const Mesh& mesh)
{
    vector<float> verts;
    vector<GLubyte>  quals;
    compileVerts(mesh, verts, quals);
    _vertElemCount = mesh.vert.size();

    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    GLuint vertSize = verts.size() * sizeof(decltype(verts.front()));
    glBufferData(GL_ARRAY_BUFFER, vertSize, verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    verts.clear();
    verts.shrink_to_fit();

    glBindBuffer(GL_ARRAY_BUFFER, _qbo);
    GLuint qualSize = quals.size() * sizeof(decltype(quals.front()));
    glBufferData(GL_ARRAY_BUFFER, qualSize, quals.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    quals.clear();
    quals.shrink_to_fit();


    vector<GLuint> edges;
    compileEdges(mesh, edges);
    _indxElemCount = edges.size();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
    GLuint edgeSize = edges.size() * sizeof(decltype(edges.front()));
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgeSize, edges.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    _buffNeedUpdate = false;
}

void ScientificRenderer::compileVerts(const Mesh& mesh, std::vector<float>& verts, std::vector<GLubyte>& quals)
{
    size_t vertCount = mesh.vert.size();

    verts.resize(vertCount * 3);
    for(int i=0, idx=-1; i < vertCount; ++i)
    {
        const glm::dvec3& v = mesh.vert[i];
        verts[++idx] = v.x;
        verts[++idx] = v.y;
        verts[++idx] = v.z;
    }


    quals.resize(vertCount, 255);

    // Tetrahedrons
    int tetCount = mesh.tetra.size();
    for(int i=0; i < tetCount; ++i)
    {
        const MeshTet& tet = mesh.tetra[i];
        GLubyte qual = 255 * _evaluator->tetrahedronQuality(mesh, tet);
        if(qual < quals[tet.v[0]]) quals[tet.v[0]] = qual;
        if(qual < quals[tet.v[1]]) quals[tet.v[1]] = qual;
        if(qual < quals[tet.v[2]]) quals[tet.v[2]] = qual;
        if(qual < quals[tet.v[3]]) quals[tet.v[3]] = qual;
    }

    // Prisms
    int priCount = mesh.prism.size();
    for(int i=0; i < priCount; ++i)
    {
        const MeshPri& pri = mesh.prism[i];
        GLubyte qual = 255 * _evaluator->prismQuality(mesh, pri);
        if(qual < quals[pri.v[0]]) quals[pri.v[0]] = qual;
        if(qual < quals[pri.v[1]]) quals[pri.v[1]] = qual;
        if(qual < quals[pri.v[2]]) quals[pri.v[2]] = qual;
        if(qual < quals[pri.v[3]]) quals[pri.v[3]] = qual;
        if(qual < quals[pri.v[4]]) quals[pri.v[4]] = qual;
        if(qual < quals[pri.v[5]]) quals[pri.v[5]] = qual;
    }

    // Hexahedrons
    int hexCount = mesh.hexa.size();
    for(int i=0; i < hexCount; ++i)
    {
        const MeshHex& hex = mesh.hexa[i];
        GLubyte qual = 255 * _evaluator->hexahedronQuality(mesh, hex);
        if(qual < quals[hex.v[0]]) quals[hex.v[0]] = qual;
        if(qual < quals[hex.v[1]]) quals[hex.v[1]] = qual;
        if(qual < quals[hex.v[2]]) quals[hex.v[2]] = qual;
        if(qual < quals[hex.v[3]]) quals[hex.v[3]] = qual;
        if(qual < quals[hex.v[4]]) quals[hex.v[4]] = qual;
        if(qual < quals[hex.v[5]]) quals[hex.v[5]] = qual;
        if(qual < quals[hex.v[6]]) quals[hex.v[6]] = qual;
        if(qual < quals[hex.v[7]]) quals[hex.v[7]] = qual;
    }
}

void ScientificRenderer::compileEdges(const Mesh& mesh, std::vector<GLuint>& edges)
{
    glm::dvec3 cutNormal(_cutPlane);
    double cutDistance = _cutPlane.w;
    if(!_isPhysicalCut)
    {
        cutNormal.x = 0;
        cutNormal.y = 0;
        cutNormal.z = 0;
        cutDistance = 0;
    }

    set<pair<GLuint, GLuint>> edgeSet;

    size_t vertCount = mesh.vertCount();
    for(int i=0; i < vertCount; ++i)
    {
        size_t neighCount = mesh.topo[i].neighbors.size();
        for(int n=0; n < neighCount; ++n)
        {
            int neig = mesh.topo[i].neighbors[n];
            if(i < neig)
                edgeSet.insert(pair<GLuint, GLuint>(i, neig));
            else
                edgeSet.insert(pair<GLuint, GLuint>(neig, i));
        }
    }

    for(const pair<int, int>& e : edgeSet)
    {
        if(glm::dot(mesh.vert[e.first].p, cutNormal) > cutDistance ||
           glm::dot(mesh.vert[e.second].p, cutNormal) > cutDistance)
            continue;

        edges.push_back(e.first);
        edges.push_back(e.second);
    }
}

void ScientificRenderer::clearResources()
{
    glDeleteVertexArrays(1, &_vao);
    _vao = 0;

    glDeleteBuffers(1, &_vbo);
    _vbo = 0;

    glDeleteBuffers(1, &_ibo);
    _ibo = 0;
}

void ScientificRenderer::resetResources()
{
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);

    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &_qbo);
    glBindBuffer(GL_ARRAY_BUFFER, _qbo);
    glVertexAttribPointer(1, 1, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}

void ScientificRenderer::setupShaders()
{
    _pointSphereProgram.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/PointSphere.vert");
    _pointSphereProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/PointSphere.frag");
    _pointSphereProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/generic/QualityLut.glsl");
    _pointSphereProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/generic/Lighting.glsl");
    _pointSphereProgram.link();
    _pointSphereProgram.pushProgram();
    _pointSphereProgram.setFloat("PointRadius", _pointRadius);
    _pointSphereProgram.setFloat("LineWidth", _lineWidth);
    _pointSphereProgram.setInt("LightMode", _lightMode);
    _pointSphereProgram.popProgram();

    _wireframeProgram.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/Wireframe.vert");
    _wireframeProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/Wireframe.frag");
    _wireframeProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/generic/QualityLut.glsl");
    _wireframeProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/generic/Lighting.glsl");
    _wireframeProgram.link();
    _wireframeProgram.pushProgram();
    _wireframeProgram.setFloat("LineRadius", _lineWidth / 2.0);
    _wireframeProgram.setInt("LightMode", _lightMode);
    _wireframeProgram.popProgram();
}

void ScientificRenderer::render()
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_PROGRAM_POINT_SIZE);

    glBindVertexArray(_vao);
    _pointSphereProgram.pushProgram();
    glDrawArrays(GL_POINTS, 0, _vertElemCount);
    _pointSphereProgram.popProgram();


    glDisable(GL_PROGRAM_POINT_SIZE);
    glLineWidth(_lineWidth);

    _wireframeProgram.pushProgram();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
    glDrawElements(GL_LINES, _indxElemCount, GL_UNSIGNED_INT, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    _wireframeProgram.popProgram();

    glLineWidth(1.0f);

    glBindVertexArray(0);
}
