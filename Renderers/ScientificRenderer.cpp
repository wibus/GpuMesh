#include "ScientificRenderer.h"

#include <iostream>

#include <GLM/gtc/matrix_transform.hpp>

#include <CellarWorkbench/Misc/Log.h>

#include <Scaena/StageManagement/Event/KeyboardEvent.h>
#include <Scaena/StageManagement/Event/SynchronousKeyboard.h>
#include <Scaena/StageManagement/Event/SynchronousMouse.h>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;
using namespace cellar;


ScientificRenderer::ScientificRenderer() :
    _vao(0),
    _vbo(0),
    _ibo(0),
    _lightMode(0),
    _tubeRadius(5.0f),
    _jointRadius(0.002f),
    _jointTubeMinRatio(1.5f),
    _isPhysicalCut(true)
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
        _projMat = glm::perspectiveFov(
                glm::pi<float>() / 6,
                (float) viewport.x,
                (float) viewport.y,
                n, f);

        glm::mat4 projInv = glm::inverse(_projMat);

        _scaffoldJointProgram.pushProgram();
        _scaffoldJointProgram.setMat4f("ProjInv", projInv);
        _scaffoldJointProgram.setMat4f("ProjMat", _projMat);
        _scaffoldJointProgram.setVec2f("Viewport", viewport);
        _scaffoldJointProgram.popProgram();

        _scaffoldTubeProgram.pushProgram();
        _scaffoldTubeProgram.setMat4f("ProjInv", projInv);
        _scaffoldTubeProgram.setMat4f("ProjMat", _projMat);
        _scaffoldTubeProgram.setVec2f("Viewport", viewport);
        _scaffoldTubeProgram.popProgram();

        _wireframeProgram.pushProgram();
        _wireframeProgram.setMat4f("ProjViewMat", _projMat * _viewMat);
        _wireframeProgram.popProgram();
    }
}

void ScientificRenderer::updateCamera(const glm::mat4& view,
                                      const glm::vec3& pos)
{
    _viewMat = view;

    _scaffoldJointProgram.pushProgram();
    _scaffoldJointProgram.setMat4f("ViewMat", _viewMat);
    _scaffoldJointProgram.setVec3f("CameraPosition", pos);
    _scaffoldJointProgram.popProgram();

    _scaffoldTubeProgram.pushProgram();
    _scaffoldTubeProgram.setMat4f("ViewMat", _viewMat);
    _scaffoldTubeProgram.setVec3f("CameraPosition", pos);
    _scaffoldTubeProgram.popProgram();

    _wireframeProgram.pushProgram();
    _wireframeProgram.setMat4f("ProjViewMat", _projMat * _viewMat);
    _wireframeProgram.popProgram();
}

void ScientificRenderer::updateLight(const glm::mat4&,
                                     const glm::vec3& pos)
{
    _scaffoldJointProgram.pushProgram();
    _scaffoldJointProgram.setVec3f("LightDirection", pos);
    _scaffoldJointProgram.popProgram();

    _scaffoldTubeProgram.pushProgram();
    _scaffoldTubeProgram.setVec3f("LightDirection", pos);
    _scaffoldTubeProgram.popProgram();
}

void ScientificRenderer::updateCutPlane(const glm::dvec4& cutEq)
{
    _cutPlane = cutEq;

    if(_isPhysicalCut)
    {
        _buffNeedUpdate = true;
    }

    _scaffoldJointProgram.pushProgram();
    _scaffoldJointProgram.setVec4f("CutPlaneEq", _cutPlane);
    _scaffoldJointProgram.popProgram();

    _scaffoldTubeProgram.pushProgram();
    _scaffoldTubeProgram.setVec4f("CutPlaneEq",_cutPlane);
    _scaffoldTubeProgram.popProgram();

    _wireframeProgram.pushProgram();
    _wireframeProgram.setVec4f("CutPlaneEq",_cutPlane);
    _wireframeProgram.popProgram();
}

void ScientificRenderer::handleKeyPress(const scaena::KeyboardEvent& event)
{
    if(event.getAscii() == 'X')
    {
        _lightMode = (_lightMode + 1) % 3;

        _scaffoldJointProgram.pushProgram();
        _scaffoldJointProgram.setInt("LightMode", _lightMode);
        _scaffoldJointProgram.popProgram();

        _scaffoldTubeProgram.pushProgram();
        _scaffoldTubeProgram.setInt("LightMode", _lightMode);
        _scaffoldTubeProgram.popProgram();

        if(_lightMode == 0)
            cout << "Using Wireframe light mode" << endl;
        else if(_lightMode == 1)
            cout << "Using Diffuse light mode" << endl;
        else if(_lightMode == 1)
            cout << "Using Phong light mode" << endl;

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

void ScientificRenderer::handleInputs(
        const scaena::SynchronousKeyboard& keyboard,
        const scaena::SynchronousMouse& mouse)
{

}

void ScientificRenderer::updateGeometry(
        const Mesh& mesh,
        const AbstractEvaluator& evaluator)
{
    // Clear old vertex attributes
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, _qbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


    // Fetch new vertex attributes
    vector<float> verts;
    vector<GLubyte>  quals;
    compileVerts(mesh, evaluator, verts, quals);
    _vertElemCount = mesh.vert.size();


    // Send new vertex attributes
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


    // Fetch new element indices
    vector<GLuint> edges;
    compileEdges(mesh, edges);
    _indxElemCount = edges.size();


    // Send new element indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
    GLuint edgeSize = edges.size() * sizeof(decltype(edges.front()));
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgeSize, edges.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    edges.clear();
    edges.shrink_to_fit();

    _buffNeedUpdate = false;
}

void ScientificRenderer::compileVerts(
        const Mesh& mesh,
        const AbstractEvaluator& evaluator,
        std::vector<float>& verts,
        std::vector<GLubyte>& quals)
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
        GLubyte qual = 255 * evaluator.tetrahedronQuality(mesh, tet);
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
        GLubyte qual = 255 * evaluator.prismQuality(mesh, pri);
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
        GLubyte qual = 255 * evaluator.hexahedronQuality(mesh, hex);
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

    glDeleteBuffers(1, &_qbo);
    _qbo = 0;

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
    // Clamp tube radius
    GLfloat lineWidthRange[2];
    glGetFloatv(GL_LINE_WIDTH_RANGE, lineWidthRange);
    float tubeMinRadus = lineWidthRange[0] / 2;
    float tubeMaxRadus = lineWidthRange[1] / 2;
    if(!(tubeMinRadus <= _tubeRadius && _tubeRadius <= tubeMaxRadus))
    {
        _tubeRadius = glm::clamp(_tubeRadius, tubeMinRadus, tubeMaxRadus);
        string log("Tube Radius clamped in range: [");
        log += to_string(tubeMinRadus) +  ", " + to_string(tubeMaxRadus) + "]";
        getLog().postMessage(new Message('W', false, log, "ScientificRenderer"));
    }

    _scaffoldJointProgram.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/ScaffoldJoint.vert");
    _scaffoldJointProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/ScaffoldJoint.frag");
    _scaffoldJointProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/generic/QualityLut.glsl");
    _scaffoldJointProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/generic/Lighting.glsl");
    _scaffoldJointProgram.link();
    _scaffoldJointProgram.pushProgram();
    _scaffoldJointProgram.setFloat("JointTubeMinRatio", _jointTubeMinRatio);
    _scaffoldJointProgram.setFloat("JointRadius", _jointRadius);
    _scaffoldJointProgram.setFloat("TubeRadius", _tubeRadius);
    _scaffoldJointProgram.setInt("LightMode", _lightMode);
    _scaffoldJointProgram.setInt("DiffuseLightMode", 1);
    _scaffoldJointProgram.popProgram();

    _scaffoldTubeProgram.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/ScaffoldTube.vert");
    _scaffoldTubeProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/ScaffoldTube.frag");
    _scaffoldTubeProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/generic/QualityLut.glsl");
    _scaffoldTubeProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/generic/Lighting.glsl");
    _scaffoldTubeProgram.link();
    _scaffoldTubeProgram.pushProgram();
    _scaffoldTubeProgram.setFloat("JointTubeMinRatio", _jointTubeMinRatio);
    _scaffoldTubeProgram.setFloat("TubeRadius", _tubeRadius);
    _scaffoldTubeProgram.setInt("LightMode", _lightMode);
    _scaffoldTubeProgram.setInt("DiffuseLightMode", 1);
    _scaffoldTubeProgram.popProgram();

    _wireframeProgram.addShader(GL_VERTEX_SHADER, ":/shaders/vertex/Wireframe.vert");
    _wireframeProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/fragment/Wireframe.frag");
    _wireframeProgram.addShader(GL_FRAGMENT_SHADER, ":/shaders/generic/QualityLut.glsl");
    _wireframeProgram.link();
}

void ScientificRenderer::render()
{
    glClearColor(0.3, 0.3, 0.3, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    if(_lightMode == 0)
    {
        glBindVertexArray(_vao);
        _wireframeProgram.pushProgram();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
        glDrawElements(GL_LINES, _indxElemCount, GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        _wireframeProgram.popProgram();
        glBindVertexArray(0);
    }
    else
    {
        glEnable(GL_PROGRAM_POINT_SIZE);

        glBindVertexArray(_vao);
        _scaffoldJointProgram.pushProgram();
        glDrawArrays(GL_POINTS, 0, _vertElemCount);
        _scaffoldJointProgram.popProgram();


        glDisable(GL_PROGRAM_POINT_SIZE);
        glLineWidth(80.0);

        _scaffoldTubeProgram.pushProgram();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
        glDrawElements(GL_LINES, _indxElemCount, GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        _scaffoldTubeProgram.popProgram();

        glLineWidth(1.0f);

        glBindVertexArray(0);
    }
}