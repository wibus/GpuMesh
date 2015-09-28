#include "ScaffoldRenderer.h"

#include <GLM/gtc/matrix_transform.hpp>

#include <CellarWorkbench/Misc/Log.h>

#include <Scaena/StageManagement/Event/KeyboardEvent.h>
#include <Scaena/StageManagement/Event/SynchronousKeyboard.h>
#include <Scaena/StageManagement/Event/SynchronousMouse.h>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;
using namespace cellar;


ScaffoldRenderer::ScaffoldRenderer() :
    _vao(0),
    _vbo(0),
    _qbo(0),
    _nibo(0),
    _eibo(0),
    _lightMode(0),
    _tubeRadius(5.0f),
    _jointRadius(0.002f),
    _jointTubeMinRatio(1.5f)
{
    _shadingFuncs.setDefault("Wireframe");
    _shadingFuncs.setContent({
        {string("Wireframe"), function<void()>(bind(&ScaffoldRenderer::useWireframeShading, this))},
        {string("Diffuse"),   function<void()>(bind(&ScaffoldRenderer::useDiffuseShading,   this))},
        {string("Phong"),     function<void()>(bind(&ScaffoldRenderer::usePhongShading,     this))},
    });
}


ScaffoldRenderer::~ScaffoldRenderer()
{
    clearResources();
}

void ScaffoldRenderer::notify(cellar::CameraMsg& msg)
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

void ScaffoldRenderer::updateCamera(const glm::mat4& view,
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

void ScaffoldRenderer::updateLight(const glm::mat4&,
                                   const glm::vec3& pos)
{
    _scaffoldJointProgram.pushProgram();
    _scaffoldJointProgram.setVec3f("LightDirection", pos);
    _scaffoldJointProgram.popProgram();

    _scaffoldTubeProgram.pushProgram();
    _scaffoldTubeProgram.setVec3f("LightDirection", pos);
    _scaffoldTubeProgram.popProgram();
}

void ScaffoldRenderer::updateCutPlane(const glm::dvec4& cutEq)
{
    _cutPlaneEq = cutEq;
    _virtualCutPlane = glm::vec4(0.0);
    _physicalCutPlane = glm::vec4(0.0);

    if(_cutType == ECutType::VirtualPlane)
    {
        _virtualCutPlane = cutEq;
    }
    else if(_cutType == ECutType::PhysicalPlane)
    {
        _virtualCutPlane = cutEq;
        _physicalCutPlane = cutEq;
        _buffNeedUpdate = true;
    }
}

void ScaffoldRenderer::handleKeyPress(const scaena::KeyboardEvent& event)
{
    AbstractRenderer::handleKeyPress(event);

    if(event.getAscii() == 'X')
    {
        _lightMode = (_lightMode + 1) % 3;

        _scaffoldJointProgram.pushProgram();
        _scaffoldJointProgram.setInt("LightMode", _lightMode);
        _scaffoldJointProgram.popProgram();

        _scaffoldTubeProgram.pushProgram();
        _scaffoldTubeProgram.setInt("LightMode", _lightMode);
        _scaffoldTubeProgram.popProgram();

        string shadingDescr = "Invalide shading mode";
        if(_lightMode == 0)
            shadingDescr = "Using Wireframe light mode";
        else if(_lightMode == 1)
            shadingDescr = "Using Diffuse light mode";
        else if(_lightMode == 1)
            shadingDescr = "Using Phong light mode";

        getLog().postMessage(new Message('I', false,
            shadingDescr, "ScaffoldRenderer"));
    }
}

void ScaffoldRenderer::handleInputs(
        const scaena::SynchronousKeyboard& keyboard,
        const scaena::SynchronousMouse& mouse)
{

}

void ScaffoldRenderer::updateGeometry(
        const Mesh& mesh,
        const AbstractEvaluator& evaluator)
{
    // Clear old vertex attributes
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, _qbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _nibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _eibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


    // Fetch new vertex attributes
    vector<float> verts;
    vector<GLubyte> quals;
    vector<GLuint> nodes;
    vector<GLuint> edges;
    compileBuffers(mesh, evaluator, verts, quals, nodes, edges);
    _vertElemCount = nodes.size();
    _indxElemCount = edges.size();


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

    // Send new element indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _nibo);
    GLuint nodeSize = nodes.size() * sizeof(decltype(nodes.front()));
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, nodeSize, nodes.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    nodes.clear();
    nodes.shrink_to_fit();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _eibo);
    GLuint edgeSize = edges.size() * sizeof(decltype(edges.front()));
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgeSize, edges.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    edges.clear();
    edges.shrink_to_fit();

    _buffNeedUpdate = false;
}

void ScaffoldRenderer::compileBuffers(
        const Mesh& mesh,
        const AbstractEvaluator& evaluator,
        std::vector<float>& verts,
        std::vector<GLubyte>& quals,
        std::vector<GLuint>& nodes,
        std::vector<GLuint>& edges) const
{
    // Vertex positions //
    size_t vertCount = mesh.verts.size();

    verts.resize(vertCount * 3);
    for(int i=0, idx=-1; i < vertCount; ++i)
    {
        const glm::dvec3& v = mesh.verts[i];
        verts[++idx] = v.x;
        verts[++idx] = v.y;
        verts[++idx] = v.z;
    }

    // Vertex qualities //
    vector<double> qualMins(vertCount, 1.0);

    // Tetrahedrons
    int tetCount = mesh.tets.size();
    for(int i=0; i < tetCount; ++i)
    {
        const MeshTet& tet = mesh.tets[i];
        double qual = evaluator.tetQuality(mesh, tet);
        if(qual < qualMins[tet.v[0]]) qualMins[tet.v[0]] = qual;
        if(qual < qualMins[tet.v[1]]) qualMins[tet.v[1]] = qual;
        if(qual < qualMins[tet.v[2]]) qualMins[tet.v[2]] = qual;
        if(qual < qualMins[tet.v[3]]) qualMins[tet.v[3]] = qual;
    }

    // Prisms
    int priCount = mesh.pris.size();
    for(int i=0; i < priCount; ++i)
    {
        const MeshPri& pri = mesh.pris[i];
        double qual = evaluator.priQuality(mesh, pri);
        if(qual < qualMins[pri.v[0]]) qualMins[pri.v[0]] = qual;
        if(qual < qualMins[pri.v[1]]) qualMins[pri.v[1]] = qual;
        if(qual < qualMins[pri.v[2]]) qualMins[pri.v[2]] = qual;
        if(qual < qualMins[pri.v[3]]) qualMins[pri.v[3]] = qual;
        if(qual < qualMins[pri.v[4]]) qualMins[pri.v[4]] = qual;
        if(qual < qualMins[pri.v[5]]) qualMins[pri.v[5]] = qual;
    }

    // Hexahedrons
    int hexCount = mesh.hexs.size();
    for(int i=0; i < hexCount; ++i)
    {
        const MeshHex& hex = mesh.hexs[i];
        double qual = evaluator.hexQuality(mesh, hex);
        if(qual < qualMins[hex.v[0]]) qualMins[hex.v[0]] = qual;
        if(qual < qualMins[hex.v[1]]) qualMins[hex.v[1]] = qual;
        if(qual < qualMins[hex.v[2]]) qualMins[hex.v[2]] = qual;
        if(qual < qualMins[hex.v[3]]) qualMins[hex.v[3]] = qual;
        if(qual < qualMins[hex.v[4]]) qualMins[hex.v[4]] = qual;
        if(qual < qualMins[hex.v[5]]) qualMins[hex.v[5]] = qual;
        if(qual < qualMins[hex.v[6]]) qualMins[hex.v[6]] = qual;
        if(qual < qualMins[hex.v[7]]) qualMins[hex.v[7]] = qual;
    }

    quals.resize(vertCount);
    if(_cutType == ECutType::InvertedElements)
    {
        for(size_t v=0; v < vertCount; ++v)
            quals[v] = 255 * glm::clamp(-qualMins[v], 0.0, 1.0);
    }
    else
    {
        for(size_t v=0; v < vertCount; ++v)
            quals[v] = 255 * glm::max(qualMins[v], 0.0);
    }


    // Build Node Visibility and Index
    vector<bool> visibility(vertCount, false);
    for(size_t v=0; v < vertCount; ++v)
    {
        if(qualMins[v] >= _qualityCullingMin &&
           qualMins[v] <= _qualityCullingMax)
        {
            for(const MeshNeigElem& n : mesh.topos[v].neighborElems)
            {
                if((_tetVisibility && n.type == MeshTet::ELEMENT_TYPE) ||
                    (_priVisibility && n.type == MeshPri::ELEMENT_TYPE) ||
                    (_hexVisibility && n.type == MeshHex::ELEMENT_TYPE))
                {
                    visibility[v] = true;
                    break;
                }
            }

            if(visibility[v])
                nodes.push_back(v);
        }
    }


    // Build Element Index
    glm::dvec3 cutNormal(_physicalCutPlane);
    double cutDistance = _physicalCutPlane.w;

    set<pair<GLuint, GLuint>> edgeSet;
    for(size_t i=0; i < vertCount; ++i)
    {
        size_t neigVertCount = mesh.topos[i].neighborVerts.size();
        for(size_t n=0; n < neigVertCount; ++n)
        {
            int neig = mesh.topos[i].neighborVerts[n];
            if(i < neig)
                edgeSet.insert(pair<GLuint, GLuint>(i, neig));
            else
                edgeSet.insert(pair<GLuint, GLuint>(neig, i));
        }
    }

    for(const pair<int, int>& e : edgeSet)
    {
        if(!visibility[e.first] ||
           !visibility[e.second])
            continue;

        if(glm::dot(mesh.verts[e.first].p, cutNormal) > cutDistance ||
           glm::dot(mesh.verts[e.second].p, cutNormal) > cutDistance)
            continue;

        if(_cutType == ECutType::InvertedElements &&
           (qualMins[e.first] > 0.0 || qualMins[e.second] > 0.0))
            continue;

        edges.push_back(e.first);
        edges.push_back(e.second);
    }
}

void ScaffoldRenderer::clearResources()
{
    glDeleteVertexArrays(1, &_vao);
    _vao = 0;

    glDeleteBuffers(1, &_vbo);
    _vbo = 0;

    glDeleteBuffers(1, &_qbo);
    _qbo = 0;

    glDeleteBuffers(1, &_nibo);
    _nibo = 0;

    glDeleteBuffers(1, &_eibo);
    _eibo = 0;
}

void ScaffoldRenderer::resetResources()
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

    glGenBuffers(1, &_nibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _nibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glGenBuffers(1, &_eibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _eibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}

void ScaffoldRenderer::setupShaders()
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
        getLog().postMessage(new Message('W', false, log, "ScaffoldRenderer"));
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

void ScaffoldRenderer::render()
{
    drawBackdrop();

    if(_lightMode == 0)
    {
        glBindVertexArray(_vao);
        _wireframeProgram.pushProgram();
        _wireframeProgram.setVec4f("CutPlaneEq", _virtualCutPlane);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _eibo);
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
        _scaffoldJointProgram.setInt("LightMode", _lightMode);
        _scaffoldJointProgram.setVec4f("CutPlaneEq", _virtualCutPlane);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _nibo);
        glDrawElements(GL_POINTS, _vertElemCount, GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        _scaffoldJointProgram.popProgram();

        glDisable(GL_PROGRAM_POINT_SIZE);
        glLineWidth(2.0 * _tubeRadius);

        _scaffoldTubeProgram.pushProgram();
        _scaffoldTubeProgram.setInt("LightMode", _lightMode);
        _scaffoldTubeProgram.setVec4f("CutPlaneEq", _virtualCutPlane);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _eibo);
        glDrawElements(GL_LINES, _indxElemCount, GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        _scaffoldTubeProgram.popProgram();

        glLineWidth(1.0f);

        glBindVertexArray(0);
    }
}

void ScaffoldRenderer::useWireframeShading()
{
    _lightMode = 0;
}

void ScaffoldRenderer::useDiffuseShading()
{
    _lightMode = 1;
}

void ScaffoldRenderer::usePhongShading()
{
    _lightMode = 2;
}
