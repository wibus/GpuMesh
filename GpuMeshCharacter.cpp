#include "GpuMeshCharacter.h"

#include <sstream>
#include <chrono>
#include <iostream>

#include <GLM/glm.hpp>
#include <GLM/gtc/random.hpp>
#include <GLM/gtc/matrix_transform.hpp>

#include <CellarWorkbench/Misc/Log.h>
#include <CellarWorkbench/Image/Image.h>
#include <CellarWorkbench/Image/ImageBank.h>
#include <CellarWorkbench/GL/GlToolkit.h>

#include <Scaena/Play/Play.h>
#include <Scaena/Play/View.h>
#include <Scaena/StageManagement/Event/KeyboardEvent.h>
#include <Scaena/StageManagement/Event/SynchronousKeyboard.h>
#include <Scaena/StageManagement/Event/SynchronousMouse.h>

using namespace std;
using namespace cellar;
using namespace scaena;



const glm::vec3 GpuMeshCharacter::nullVec = glm::vec3(0, 0, 0);
const glm::vec3 GpuMeshCharacter::upVec = glm::vec3(0, 0, 1);

GpuMeshCharacter::GpuMeshCharacter() :
    Character("GpuMeshChracter"),
    _buffElemCount(0),
    _vao(0),
    _vbo(0),
    _nbo(0),
    _ebo(0),
    _qbo(0),
    _shadowFbo(0),
    _shadowDpt(0),
    _shadowTex(0),
    _useBackdrop(true),
    _backdropTex(0),
    _backdropVao(0),
    _backdropVbo(0),
    _useLitShader(true),
    _shadowEnabled(true),
    _updateShadow(false),
    _shadowSize(1024, 1024),
    _camAzimuth(0),
    _camAltitude(0),
    _camDistance(6),
    _isPhysicalCut(true),
    _cutAzimuth(0),
    _cutAltitude(0),
    _cutDistance(0),
    _lightAzimuth(-glm::pi<float>() * 3.5 / 8.0),
    _lightAltitude(-glm::pi<float>() * 1.0 / 4.0),
    _lightDistance(1.0),
    _internalVertices(1000),
    _useGpuPipeline(false),
    _processFinished(false),
    _stepId(0)
{
}

void GpuMeshCharacter::enterStage()
{
    setupShaders();
    resetCpuPipeline();
    resetGpuPipeline();

    play().view()->camera3D()->registerObserver(*this);

    _processFinished = false;
    _stepId = 0;
}

void GpuMeshCharacter::beginStep(const StageTime &time)
{
    std::shared_ptr<SynchronousKeyboard> keyboard = play().synchronousKeyboard();
    std::shared_ptr<SynchronousMouse> mouse = play().synchronousMouse();

    if(!_processFinished)
    {
        auto startTime = chrono::high_resolution_clock::now();

        if(_useGpuPipeline)
        {
            processGpuPipeline();
        }
        else
        {
            processCpuPipeline();
        }

        auto endTime = chrono::high_resolution_clock::now();
        auto dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

        stringstream ss;
        ss << "Step took " << dt.count() / 1000.0 << "ms to execute";
        getLog().postMessage(new Message('I', false, ss.str(), "GpuMeshCharacter"));
    }


    if(keyboard->isNonAsciiPressed(ENonAscii::SHIFT))
    {
        // Cut plane management
        if(mouse->buttonIsPressed(EMouseButton::LEFT))
        {
            moveCutPlane(_cutAzimuth - mouse->displacement().x / 200.0f,
                         _cutAltitude - mouse->displacement().y / 200.0f,
                         _cutDistance);
        }
        else if(mouse->degreeDelta() != 0)
        {
            moveCutPlane(_cutAzimuth,
                         _cutAltitude,
                         _cutDistance + mouse->degreeDelta() / 800.0f);
        }
    }
    else if(keyboard->isNonAsciiPressed(ENonAscii::CTRL))
    {
        // Cut plane management
        if(mouse->buttonIsPressed(EMouseButton::LEFT))
        {
            moveLight(_lightAzimuth - mouse->displacement().x / 200.0f,
                      _lightAltitude - mouse->displacement().y / 200.0f,
                      _lightDistance);
        }
        else if(mouse->degreeDelta() != 0)
        {
            moveLight(_lightAzimuth,
                      _lightAltitude,
                      _lightDistance + mouse->degreeDelta() / 800.0f);
        }
    }
    else
    {
        // Camera management
        if(mouse->buttonIsPressed(EMouseButton::LEFT))
        {
            moveCamera(_camAzimuth - mouse->displacement().x / 100.0f,
                       _camAltitude - mouse->displacement().y / 100.0f,
                       _camDistance);
        }
        else if(mouse->degreeDelta() != 0)
        {
            moveCamera(_camAzimuth,
                       _camAltitude,
                       _camDistance + mouse->degreeDelta() / 80.0f);
        }
    }
}

void GpuMeshCharacter::draw(const shared_ptr<View>&, const StageTime&)
{
    glBindVertexArray(_vao);
    glBindTexture(GL_TEXTURE_2D, _shadowTex);


    // Render shadow map
    if(_shadowEnabled && _updateShadow)
    {
        int viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        glBindFramebuffer(GL_FRAMEBUFFER, _shadowFbo);
        glViewport(0, 0, _shadowSize.x, _shadowSize.y);

        glClearColor(1.0, 1.0, 1.0, 1.0);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        _shadowShader.pushProgram();
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glDisableVertexAttribArray(3);
        glDrawArrays(GL_TRIANGLES, 0, _buffElemCount);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glEnableVertexAttribArray(3);
        _shadowShader.popProgram();

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(viewport[0], viewport[1],
                   viewport[2], viewport[3]);

        glGenerateMipmap(GL_TEXTURE_2D);

        _updateShadow = false;
    }


    // Render background
    if(_useBackdrop)
    {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, _backdropTex);
        glActiveTexture(GL_TEXTURE0);

        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        _backdropShader.pushProgram();
        glBindVertexArray(_backdropVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(_vao);
        _backdropShader.popProgram();

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
    }
    else
    {
        glClearColor(0.93, 0.95, 1.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
    }


    // Render mesh
    if(_useLitShader)
        _litShader.pushProgram();
    else
        _unlitShader.pushProgram();
    glDrawArrays(GL_TRIANGLES, 0, _buffElemCount);
    GlProgram::popProgram();
    glBindVertexArray(0);
}

void GpuMeshCharacter::exitStage()
{
}

bool GpuMeshCharacter::keyPressEvent(const scaena::KeyboardEvent &event)
{
    if(_processFinished)
    {
        if(event.getAscii()  == 'S')
        {
            _processFinished = false;
        }
        else if(event.getAscii() == 'Z')
        {
            _useLitShader = !_useLitShader;
            cout << "Using lit shader : " << (_useLitShader ? "true" : "false") << endl;
        }
        else if(event.getAscii() == 'X')
        {
            _shadowEnabled = !_shadowEnabled;
            cout << "Shadow enabled : " << (_shadowEnabled ? "true" : "false") << endl;

            if(_shadowEnabled)
            {
                _updateShadow = true;
            }
            else
            {
                glBindFramebuffer(GL_FRAMEBUFFER, _shadowFbo);
                glClearColor(1.0, 1.0, 1.0, 1.0);
                glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glBindTexture(GL_TEXTURE_2D, _shadowTex);
                glGenerateMipmap(GL_TEXTURE_2D);
            }
        }
        else if(event.getAscii() == 'C')
        {
            _isPhysicalCut = !_isPhysicalCut;
            cout << "Physical cut : " << (_isPhysicalCut ? "true" : "false") << endl;

            moveCutPlane(_cutAzimuth, _cutAltitude, _cutDistance);

            if(!_isPhysicalCut)
            {
                updateBuffers();
            }
        }
    }
}

void GpuMeshCharacter::notify(CameraMsg& msg)
{
    if(msg.change == CameraMsg::EChange::VIEWPORT)
    {
        const glm::ivec2& viewport = msg.camera.viewport();

        // Camera projection
        _camProj = glm::perspectiveFov(
                glm::pi<float>() / 6,
                (float) viewport.x,
                (float) viewport.y,
                0.1f,
                12.0f);
        moveCamera(_camAzimuth,
                   _camAltitude,
                   _camDistance);

        // Background scale
        glm::vec2 scale(1.0);
        glm::vec2 viewportf(viewport);
        glm::vec2 backSize(_backdropWidth, _backdropHeight);
        if(glm::any(glm::greaterThan(viewportf, backSize)))
        {
            glm::vec2 ratio = viewportf / backSize;
            if(ratio.x < ratio.y)
                scale.x = 1.0f / ratio.y;
            else
                scale.y = 1.0f / ratio.x;
        }
        else
            scale = viewportf / backSize;

        _backdropShader.pushProgram();
        _backdropShader.setVec2f("TexScale", scale);
        _backdropShader.popProgram();

        _updateShadow = true;
    }
}

void GpuMeshCharacter::processCpuPipeline()
{
    switch(_stepId)
    {
    case 0:
        printStep(_stepId, "Generation of boundary surfaces");
        genBoundaryMeshesCpu();
        ++_stepId;
        break;

    case 1:
        printStep(_stepId, "Triangulation of internal domain");
        triangulateDomainCpu();

        _processFinished = true;
        ++_stepId;
        break;

    case 2:
        printStep(_stepId, "Computing adjacency lists");
        computeAdjacencyCpu();
        ++_stepId;
        break;

    case 3:
        printStep(_stepId, "Smoothing of the internal domain");
        smoothMeshCpu();

        _processFinished = true;
        break;

    default:
        _processFinished = true;
        getLog().postMessage(new Message(
            'E', false, "Invalid step", "GpuMeshCharacter"));
    }
}

void GpuMeshCharacter::genBoundaryMeshesCpu()
{
    double a = 20.0;

    vector<glm::dvec3> vertices;
    vertices.push_back(glm::dvec3(-a, -a,  a));
    vertices.push_back(glm::dvec3( a, -a,  a));
    vertices.push_back(glm::dvec3(-a,  a,  a));
    vertices.push_back(glm::dvec3( a,  a,  a));
    vertices.push_back(glm::dvec3(-a, -a, -a));
    vertices.push_back(glm::dvec3( a, -a, -a));
    vertices.push_back(glm::dvec3(-a,  a, -a));
    vertices.push_back(glm::dvec3( a,  a, -a));

    std::vector<Tetrahedron> tetrahedron;
    tetrahedron.push_back(Tetrahedron(0, 1, 2, 4));
    tetrahedron.push_back(Tetrahedron(5, 4, 7, 1));
    tetrahedron.push_back(Tetrahedron(3, 1, 7, 2));
    tetrahedron.push_back(Tetrahedron(6, 2, 7, 4));
    tetrahedron.push_back(Tetrahedron(4, 1, 2, 7));

    _mesh.initialize(vertices, tetrahedron);
    updateBuffers();
}

void GpuMeshCharacter::triangulateDomainCpu()
{
    chrono::high_resolution_clock::time_point startTime, endTime;
    chrono::microseconds dt;

    double sphereRadius = 1.0;
    glm::dvec3 cMin(-sphereRadius);
    glm::dvec3 cMax( sphereRadius);
    std::vector<glm::dvec3> vertices;


    /* Box distribution
    vertices.resize(_internalVertices);

    int surfCount = glm::sqrt(_internalVertices) * 10;
    int padding = glm::pow(_internalVertices, 1/3.0);
    for(int iv=0; iv < surfCount; ++iv)
    {
        glm::dvec3 val = glm::linearRand(cMin, cMax);
        val[iv%3] = glm::mix(-1.0, 1.0, (double)(iv%2));
        vertices[iv] = val;
    }

    for(int iv=surfCount; iv<_internalVertices; ++iv)
        vertices[iv] = glm::linearRand(cMin, cMax) * (1.0 - 1.0 / padding);

    startTime = chrono::high_resolution_clock::now();
    _mesh.insertVertices(vertices);
    endTime = chrono::high_resolution_clock::now();

    dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Total meshing time = " << dt.count() / 1000.0 << "ms" << endl;
    //*/

    //* Sphere distribution
    vertices.resize(_internalVertices);

    for(int iv=0; iv<_internalVertices; ++iv)
        vertices[iv] = glm::ballRand(sphereRadius * 1.41);

    startTime = chrono::high_resolution_clock::now();
    _mesh.insertVertices(vertices);
    endTime = chrono::high_resolution_clock::now();

    dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Total meshing time = " << dt.count() / 1000.0 << "ms" << endl;
    //*/


    /* Paraboloid cap
    int stackCount = glm::pow(_internalVertices/2.3, 1/3.0);
    glm::dvec3 minPerturb(-0.1 / stackCount);
    glm::dvec3 maxPerturb(0.1 / stackCount);
    for(int s=0; s <= stackCount; ++s)
    {
        double sProg = s / (double) stackCount;
        double height = sProg;
        vertices.push_back(glm::dvec3(0, 0, height));

        for(int r=1; r <= s; ++r)
        {
            double rProg = r / (double) s;
            double ringRadius = r / (double) stackCount;
            double ringHeight = height * (1.0 - rProg*rProg);
            double ringVertCount = stackCount * glm::sqrt(1.0 + r);
            for(int v=0; v < ringVertCount; ++v)
            {
                double vProg = v / (double) ringVertCount;
                double vAngle = glm::pi<double>() * vProg * 2;
                vertices.push_back(glm::dvec3(
                    glm::cos(vAngle) * ringRadius,
                    glm::sin(vAngle) * ringRadius * 0.75,
                    ringHeight)
                                   +
                    glm::linearRand(minPerturb, maxPerturb)
                     * (1.1 - rProg) * (1.1 - sProg));
            }
        }
    }

    startTime = chrono::high_resolution_clock::now();
    _mesh.insertVertices(vertices);
    endTime = chrono::high_resolution_clock::now();

    dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Total meshing time = " << dt.count() / 1000.0 << "ms" << endl;
    //*/


    //exit(0);
    updateBuffers();
}

void GpuMeshCharacter::computeAdjacencyCpu()
{
    _mesh.compileAdjacencyLists();
}

void GpuMeshCharacter::smoothMeshCpu()
{
    const glm::dvec3 MOVE_FACTOR(0.5);
    const double SMOOTH_AMELIORATION_THRESHOLD = 0.001;
    double dQuality = 1.0;
    int smoothPass = 0;

    while(dQuality > SMOOTH_AMELIORATION_THRESHOLD)
    {
        cout << "Smooth pass number" << smoothPass << endl;
        cout << "Input mesh quality mean: " << _mesh.qualityMean << endl;
        cout << "Input mesh quality std dev: " << _mesh.qualityVar << endl;
        double lastQualityMean = _mesh.qualityMean;

        int vertCount = _mesh.vertCount();
        int firstVert = _mesh.externalVertCount;
        for(int v = firstVert; v < vertCount; ++v)
        {
            if(_mesh.vert[v].isBoundary)
                continue;

            double weightSum = 0.0;
            glm::dvec3 barycenter;

            glm::dvec3& vertPos = _mesh.vert[v].p;
            for(auto& n : _mesh.neighbors[v])
            {
                const glm::dvec3& neighborPos = _mesh.vert[n].p;
                glm::dvec3 dist = vertPos - neighborPos;
                double weight = glm::log(glm::dot(dist, dist) + 1);

                barycenter = (barycenter * weightSum + neighborPos * weight)
                              / (weightSum + weight);
                weightSum += weight;
            }

            vertPos = glm::mix(vertPos, barycenter, MOVE_FACTOR);
        }

        updateBuffers();

        dQuality = _mesh.qualityMean - lastQualityMean;
        ++smoothPass;
    }

    cout << "#Smoothing finished" << endl;
    cout << "Final mesh quality mean: " << _mesh.qualityMean << endl;
    cout << "Final mesh quality std dev: " << _mesh.qualityVar << endl << endl;
}


// GPU Pipeline
void GpuMeshCharacter::resetGpuPipeline()
{

}

void GpuMeshCharacter::processGpuPipeline()
{
    processCpuPipeline();
}


// Common management
void GpuMeshCharacter::printStep(int step, const std::string& stepName)
{
    stringstream ss;
    ss << "Step " << step << ": Executing " << stepName;
    getLog().postMessage(new Message('I', false, ss.str(), "GpuMeshCharacter"));
}

void GpuMeshCharacter::moveCamera(float azimuth, float altitude, float distance)
{
    const float PI = glm::pi<float>();
    _camAzimuth = glm::mod(azimuth, 2 * PI);
    _camAltitude = glm::clamp(altitude, -PI * 0.48f, PI * 0.48f);
    _camDistance = glm::clamp(distance, 0.05f, 10.0f);

    glm::vec3 from = glm::vec3(
            glm::rotate(glm::mat4(), _camAzimuth, glm::vec3(0, 0, 1)) *
            glm::rotate(glm::mat4(), _camAltitude, glm::vec3(0, 1, 0)) *
            glm::vec4(_camDistance, 0.0f, 0.0f, 1.0f));

    glm::mat4 viewMatrix = glm::lookAt(from, nullVec, upVec);
    glm::mat4 pvMat = _camProj * viewMatrix;
    glm::vec3 camPos(from);

    _litShader.pushProgram();
    _litShader.setMat4f("PVmat", pvMat);
    _litShader.setVec3f("CameraPosition", camPos);
    _litShader.popProgram();

    _unlitShader.pushProgram();
    _unlitShader.setMat4f("PVmat", pvMat);
    _unlitShader.setVec3f("CameraPosition", camPos);
    _unlitShader.popProgram();
}

void GpuMeshCharacter::moveCutPlane(float azimuth, float altitude, float distance)
{
    const float PI = glm::pi<float>();
    _cutAzimuth = glm::mod(azimuth, 2 * PI);
    _cutAltitude = glm::clamp(altitude, -PI / 2, PI / 2);
    _cutDistance = glm::clamp(distance, -2.0f, 2.0f);

    glm::vec4 cutPlaneEq =
            glm::rotate(glm::mat4(), _cutAzimuth, glm::vec3(0, 0, 1)) *
            glm::rotate(glm::mat4(), _cutAltitude, glm::vec3(0, 1, 0)) *
            glm::vec4(1.0, 0.0f, 0.0f, 1.0f);
    cutPlaneEq.w = _cutDistance;

    glm::dvec4 virtualCut(0, 0, 0, 0);

    if(_isPhysicalCut)
    {
        _physicalCutPlaneEq = cutPlaneEq;
        updateBuffers();
    }
    else
    {
        _physicalCutPlaneEq = virtualCut;
        virtualCut = cutPlaneEq;
    }


    _litShader.pushProgram();
    _litShader.setVec4f("CutPlaneEq", virtualCut);
    _litShader.popProgram();

    _unlitShader.pushProgram();
    _unlitShader.setVec4f("CutPlaneEq",virtualCut);
    _unlitShader.popProgram();

    _shadowShader.pushProgram();
    _shadowShader.setVec4f("CutPlaneEq", virtualCut);
    _shadowShader.popProgram();

    _updateShadow = true;
}


void GpuMeshCharacter::moveLight(float azimuth, float altitude, float distance)
{
    const float PI = glm::pi<float>();
    _lightAzimuth = glm::mod(azimuth, 2 *PI );
    _lightAltitude = glm::clamp(altitude, -PI / 2, PI / 2);
    _lightDistance = glm::clamp(distance, 1.0f, 10.0f);

    glm::vec3 from = glm::vec3(
            glm::rotate(glm::mat4(), _lightAzimuth, glm::vec3(0, 0, 1)) *
            glm::rotate(glm::mat4(), _lightAltitude, glm::vec3(0, 1, 0)) *
            glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

    glm::mat4 view = glm::lookAt(nullVec, -from , upVec);
    glm::mat4 pvMat = _shadowProj * view;
    glm::mat4 pvShadow =
            glm::scale(glm::mat4(), glm::vec3(0.5)) *
            glm::translate(glm::mat4(), glm::vec3(1.0, 1.0, 1.0)) *
            pvMat;

    _litShader.pushProgram();
    _litShader.setVec3f("LightDirection", -from);
    _litShader.setMat4f("PVshadow", pvShadow);
    _litShader.popProgram();

    _unlitShader.pushProgram();
    _unlitShader.setVec3f("LightDirection", -from);
    _unlitShader.popProgram();

    _shadowShader.pushProgram();
    _shadowShader.setMat4f("PVmat", pvMat);
    _shadowShader.popProgram();

    _updateShadow = true;
}

void GpuMeshCharacter::setupShaders()
{
    // Compile shaders
    _litShader.addShader(GL_VERTEX_SHADER, ":/shaders/LitMesh.vert");
    _litShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/LitMesh.frag");
    _litShader.link();
    _litShader.pushProgram();
    _litShader.setInt("DepthTex", 0);
    _litShader.popProgram();

    _unlitShader.addShader(GL_VERTEX_SHADER, ":/shaders/UnlitMesh.vert");
    _unlitShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/UnlitMesh.frag");
    _unlitShader.link();
    _unlitShader.pushProgram();
    _unlitShader.popProgram();

    _shadowShader.addShader(GL_VERTEX_SHADER, ":/shaders/Shadow.vert");
    _shadowShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/Shadow.frag");
    _shadowShader.link();
    _shadowShader.pushProgram();
    _shadowShader.popProgram();

    _backdropShader.addShader(GL_VERTEX_SHADER, ":/shaders/Backdrop.vert");
    _backdropShader.addShader(GL_FRAGMENT_SHADER, ":/shaders/Backdrop.frag");
    _backdropShader.link();
    _backdropShader.pushProgram();
    _backdropShader.setInt("Backdrop", 1);
    _backdropShader.setVec2f("TexScale", glm::vec2(1.0f));
    _backdropShader.popProgram();


    // Set shadow projection view matrix
    _shadowProj = glm::ortho(
            -2.0f, 2.0f,
            -2.0f, 2.0f,
            -2.0f, 2.0f);

    // Setup view matrix
    moveCamera(_camAzimuth, _camAltitude, _camDistance);

    // Setup cut plane
    moveCutPlane(_cutAzimuth, _cutAltitude, _cutDistance);

    // Setup shadow matrix
    moveLight(_lightAzimuth, _lightAltitude, _lightDistance);
}

void GpuMeshCharacter::updateBuffers()
{
    vector<glm::vec3> vertices;
    vector<glm::vec3> normals;
    vector<glm::vec3> edges;
    vector<float> qualities;

    _mesh.compileFacesAttributes(
                _physicalCutPlaneEq,
                vertices,
                normals,
                edges,
                qualities);

    glBindVertexArray(_vao);

    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    GLuint verticesSize = vertices.size() * sizeof(decltype(vertices.front()));
    glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, _nbo);
    GLuint normalsSize = normals.size() * sizeof(decltype(normals.front()));
    glBufferData(GL_ARRAY_BUFFER, normalsSize, normals.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, _ebo);
    GLuint edgesSize = edges.size() * sizeof(decltype(edges.front()));
    glBufferData(GL_ARRAY_BUFFER, edgesSize, edges.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, _qbo);
    GLuint qualitiesSize = qualities.size() * sizeof(decltype(qualities.front()));
    glBufferData(GL_ARRAY_BUFFER, qualitiesSize, qualities.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);

    _buffElemCount = vertices.size();
}


// CPU Pipeline
void GpuMeshCharacter::resetCpuPipeline()
{
    // Delete old buffers
    glDeleteVertexArrays(1, &_vao);
    _vao = 0;

    glDeleteBuffers(1, &_vbo);
    _vbo = 0;

    glDeleteBuffers(1, &_nbo);
    _nbo = 0;

    glDeleteBuffers(1, &_ebo);
    _ebo = 0;

    glDeleteBuffers(1, &_qbo);
    _qbo = 0;

    GlToolkit::deleteTextureId(_backdropTex);
    _backdropTex = 0;

    glDeleteBuffers(1, &_backdropVao);
    _backdropVao = 0;

    glDeleteBuffers(1, &_backdropVbo);
    _backdropVbo = 0;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteRenderbuffers(1, &_shadowDpt);
    _shadowDpt = 0;

    glDeleteFramebuffers(1, &_shadowFbo);
    _shadowFbo = 0;

    glDeleteTextures(1, &_shadowTex);
    _shadowTex = 0;

    _buffElemCount = 0;


    // Generate new buffers
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);

    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &_nbo);
    glBindBuffer(GL_ARRAY_BUFFER, _nbo);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &_ebo);
    glBindBuffer(GL_ARRAY_BUFFER, _ebo);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &_qbo);
    glBindBuffer(GL_ARRAY_BUFFER, _qbo);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);


    // Shadow casting
    glGenTextures(1, &_shadowTex);
    glBindTexture(GL_TEXTURE_2D, _shadowTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, _shadowSize.x, _shadowSize.y,
                 0, GL_RG, GL_UNSIGNED_INT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenRenderbuffers(1, &_shadowDpt);
    glBindRenderbuffer(GL_RENDERBUFFER, _shadowDpt);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, _shadowSize.x, _shadowSize.y);

    glGenFramebuffers(1, &_shadowFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _shadowFbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _shadowDpt);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _shadowTex, 0);

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);


    // Backdrop
    GLfloat backTriangle[] = {
        -1.0f, -1.0f,
         3.0f, -1.0f,
        -1.0f,  3.0f
    };

    Image& backdrop =  getImageBank().getImage("resources/textures/Backdrop.png");
    _backdropTex = GlToolkit::genTextureId(backdrop);
    _backdropWidth = backdrop.width();
    _backdropHeight = backdrop.height();

    glGenVertexArrays(1, &_backdropVao);
    glBindVertexArray(_backdropVao);

    glGenBuffers(1, &_backdropVbo);
    glBindBuffer(GL_ARRAY_BUFFER, _backdropVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(backTriangle), backTriangle, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}
