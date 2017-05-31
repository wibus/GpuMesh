#include "SurfacicRenderer.h"

#include <GLM/gtc/matrix_transform.hpp>

#include <Scaena/StageManagement/Event/KeyboardEvent.h>
#include <Scaena/StageManagement/Event/SynchronousKeyboard.h>
#include <Scaena/StageManagement/Event/SynchronousMouse.h>

using namespace std;
using namespace cellar;


SurfacicRenderer::SurfacicRenderer() :
    _buffFaceElemCount(0),
    _buffEdgeIdxCount(0),
    _faceVao(0),
    _faceVbo(0),
    _faceNbo(0),
    _faceQbo(0),
    _shadowFbo(0),
    _shadowDpt(0),
    _shadowTex(0),
    _bloomFbo(0),
    _bloomDpt(0),
    _bloomBaseTex(0),
    _bloomBlurTex(0),
    _lightingEnabled(false),
    _updateShadow(false),
    _shadowSize(1024, 1024)
{
    _shadingFuncs.setDefault("Diffuse");
    _shadingFuncs.setContent({
        {string("Diffuse"),  function<void()>(bind(&SurfacicRenderer::useDiffuseShading,  this))},
        {string("Specular"), function<void()>(bind(&SurfacicRenderer::useSpecularShading, this))},
    });
}

SurfacicRenderer::~SurfacicRenderer()
{
    clearResources();
}

void SurfacicRenderer::updateCamera(const glm::vec3& pos)
{
    _litShader.pushProgram();
    _litShader.setVec3f("CameraPosition", pos);
    _litShader.popProgram();

    _unlitShader.pushProgram();
    _unlitShader.setVec3f("CameraPosition", pos);
    _unlitShader.popProgram();

    _edgeShader.pushProgram();
    _edgeShader.setVec3f("CameraPosition", pos);
    _edgeShader.popProgram();
}

void SurfacicRenderer::updateLight(const glm::mat4& view,
                                   const glm::vec3& pos)
{
    glm::mat4 pvMat = _shadowProj * view;

    glm::mat4 pvShadow =
            glm::scale(glm::mat4(),     glm::vec3(0.5, 0.5, 0.5)) *
            glm::translate(glm::mat4(), glm::vec3(1.0, 1.0, 1.0)) *
            pvMat;


    _litShader.pushProgram();
    _litShader.setVec3f("LightDirection", pos);
    _litShader.setMat4f("PVshadow", pvShadow);
    _litShader.popProgram();

    _unlitShader.pushProgram();
    _unlitShader.setVec3f("LightDirection", pos);
    _unlitShader.popProgram();

    _shadowShader.pushProgram();
    _shadowShader.setMat4f("PVmat", pvMat);
    _shadowShader.popProgram();

    _updateShadow = true;
}

void SurfacicRenderer::updateCutPlane(const glm::dvec4& cutEq)
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

    _updateShadow = true;
}

bool SurfacicRenderer::handleKeyPress(const scaena::KeyboardEvent& event)
{
    AbstractRenderer::handleKeyPress(event);

    if(event.getAscii() == 'X')
    {
        if(_lightingEnabled)
            useDiffuseShading();
        else
            useSpecularShading();

        const char* rep = (_lightingEnabled ? "true" : "false");
        getLog().postMessage(new Message('I', false,
            string("Lighting enabled : ") + rep, "ScaffoldRenderer"));
		return true;
    }

	return false;
}

void SurfacicRenderer::handleInputs(const scaena::SynchronousKeyboard& keyboard,
                                    const scaena::SynchronousMouse& mouse)
{

}

void SurfacicRenderer::notifyCameraUpdate(cellar::CameraMsg& msg)
{
    if(msg.change == CameraMsg::EChange::VIEWPORT)
    {
        const glm::ivec2& viewport = msg.camera.viewport();

        // Effects scale
        glm::vec2 scale = filterScale();

        _screenShader.pushProgram();
        _screenShader.setVec2f("TexScale", scale);
        _screenShader.popProgram();

        _brushShader.pushProgram();
        _brushShader.setVec2f("TexScale", scale);
        _brushShader.popProgram();

        _grainShader.pushProgram();
        _grainShader.setVec2f("TexScale", scale);
        _grainShader.popProgram();

        _updateShadow = true;

        // Resize bloom buffers
        glBindTexture(GL_TEXTURE_2D, _bloomBaseTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, viewport.x, viewport.y,
                     0, GL_RGB, GL_UNSIGNED_INT, NULL);

        glBindTexture(GL_TEXTURE_2D, _bloomBlurTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, viewport.x, viewport.y,
                     0, GL_RGB, GL_UNSIGNED_INT, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, _bloomDpt);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32,
                              viewport.x, viewport.y);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
    else if(msg.change == CameraMsg::EChange::PROJECTION)
    {
        const glm::mat4& proj = msg.camera.projectionMatrix();

        _litShader.pushProgram();
        _litShader.setMat4f("ProjMat", proj);
        _litShader.popProgram();

        _unlitShader.pushProgram();
        _unlitShader.setMat4f("ProjMat", proj);
        _unlitShader.popProgram();

        _edgeShader.pushProgram();
        _edgeShader.setMat4f("ProjMat", proj);
        _edgeShader.popProgram();
    }
    else if(msg.change == CameraMsg::EChange::VIEW)
    {
        const glm::mat4& view = msg.camera.viewMatrix();

        _litShader.pushProgram();
        _litShader.setMat4f("ViewMat", view);
        _litShader.popProgram();

        _unlitShader.pushProgram();
        _unlitShader.setMat4f("ViewMat", view);
        _unlitShader.popProgram();

        _edgeShader.pushProgram();
        _edgeShader.setMat4f("ViewMat", view);
        _edgeShader.popProgram();
    }
}

void SurfacicRenderer::updateGeometry(const Mesh& mesh)
{
    // Clear old vertex attributes
    glBindBuffer(GL_ARRAY_BUFFER, _faceVbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, _faceNbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, _faceQbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, _edgeVbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, _edgeIbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);


    // Fetch new vertex attributes
    vector<glm::vec3> faceVertices;
    vector<signed char> normals;
    vector<unsigned char> qualities;
    vector<glm::vec3> edgeVertices;
    vector<GLuint> edgeIndices;

    compileMeshAttributes(
        mesh,
        faceVertices,
        normals,
        qualities,
        edgeVertices,
        edgeIndices);


    _buffFaceElemCount = faceVertices.size();
    _buffEdgeIdxCount = edgeIndices.size();


    // Send new vertex attributes
    glBindBuffer(GL_ARRAY_BUFFER, _faceVbo);
    GLuint faceVerticesSize = faceVertices.size() * sizeof(decltype(faceVertices.front()));
    glBufferData(GL_ARRAY_BUFFER, faceVerticesSize, faceVertices.data(), GL_STATIC_DRAW);
    faceVertices.clear();
    faceVertices.shrink_to_fit();

    glBindBuffer(GL_ARRAY_BUFFER, _faceNbo);
    GLuint normalsSize = normals.size() * sizeof(decltype(normals.front()));
    glBufferData(GL_ARRAY_BUFFER, normalsSize, normals.data(), GL_STATIC_DRAW);
    normals.clear();
    normals.shrink_to_fit();

    glBindBuffer(GL_ARRAY_BUFFER, _faceQbo);
    GLuint qualitiesSize = qualities.size() * sizeof(decltype(qualities.front()));
    glBufferData(GL_ARRAY_BUFFER, qualitiesSize, qualities.data(), GL_STATIC_DRAW);
    qualities.clear();
    qualities.shrink_to_fit();

    glBindBuffer(GL_ARRAY_BUFFER, _edgeVbo);
    GLuint edgeVerticesSize = edgeVertices.size() * sizeof(decltype(edgeVertices.front()));
    glBufferData(GL_ARRAY_BUFFER, edgeVerticesSize, edgeVertices.data(), GL_STATIC_DRAW);
    edgeVertices.clear();
    edgeVertices.shrink_to_fit();

    glBindBuffer(GL_ARRAY_BUFFER, _edgeIbo);
    GLuint edgIndicesSize = edgeIndices.size() * sizeof(decltype(edgeIndices.front()));
    glBufferData(GL_ARRAY_BUFFER, edgIndicesSize, edgeIndices.data(), GL_STATIC_DRAW);
    edgeIndices.clear();
    edgeIndices.shrink_to_fit();


    glBindBuffer(GL_ARRAY_BUFFER, 0);

    _buffNeedUpdate = false;
}

void SurfacicRenderer::compileMeshAttributes(
        const Mesh& mesh,
        std::vector<glm::vec3>& faceVertices,
        std::vector<signed char>& normals,
        std::vector<unsigned char>& qualities,
        std::vector<glm::vec3>& edgeVertices,
        std::vector<GLuint>& edgeIndices) const
{
    size_t vertCount = mesh.verts.size();
    glm::dvec3 cutNormal(_physicalCutPlane);
    double cutDistance = _physicalCutPlane.w;

    set<pair<GLuint, GLuint>> edgeSet;

    // Tetrahedrons
    if(_tetVisibility)
    {
        int tetCount = mesh.tets.size();
        for(int i=0; i < tetCount; ++i)
        {
            const MeshTet& tet = mesh.tets[i];

            glm::dvec3 verts[] = {
                glm::dvec3(mesh.verts[tet[0]]),
                glm::dvec3(mesh.verts[tet[1]]),
                glm::dvec3(mesh.verts[tet[2]]),
                glm::dvec3(mesh.verts[tet[3]])
            };

            if(_cutType == ECutType::PhysicalPlane)
            {
                if(glm::dot(verts[0], cutNormal) > cutDistance ||
                   glm::dot(verts[1], cutNormal) > cutDistance ||
                   glm::dot(verts[2], cutNormal) > cutDistance ||
                   glm::dot(verts[3], cutNormal) > cutDistance)
                    continue;
            }


            double quality = tet.value;
            if(quality >= _qualityCullingMin &&
               quality <= _qualityCullingMax)
            {
                if(_cutType == ECutType::InvertedElements)
                {
                    if(quality >= 0.0)
                        continue;
                    quality = glm::min(-quality, 1.0);
                }
                else
                {
                    quality = glm::max(quality, 0.0);
                }

                for(int f=0; f < MeshTet::TRI_COUNT; ++f)
                {
                    const MeshTri& tri = MeshTet::tris[f];
                    glm::dvec3 A = verts[tri[1]] - verts[tri[0]];
                    glm::dvec3 B = verts[tri[2]] - verts[tri[1]];
                    glm::dvec3 normal = glm::normalize(glm::cross(A, B));
                    pushTriangle(faceVertices, normals, qualities,
                                 verts[tri[0]], verts[tri[1]], verts[tri[2]],
                                 normal, quality);
                }

                // Add verts to edge
                for(uint e=0; e < MeshTet::EDGE_COUNT; ++e)
                {
                    MeshEdge local = MeshTet::edges[e];
                    MeshEdge global(tet.v[local.v[0]], tet.v[local.v[1]]);
                    if(global.v[0] < global.v[1])
                        edgeSet.insert(pair<GLuint, GLuint>(global.v[0], global.v[1]));
                    else
                        edgeSet.insert(pair<GLuint, GLuint>(global.v[1], global.v[0]));
                }
            }
        }
    }


    // Pyramids
    if(_pyrVisibility)
    {
        int pyrCount = mesh.pyrs.size();
        for(int i=0; i < pyrCount; ++i)
        {
            const MeshPyr& pyr = mesh.pyrs[i];

            glm::dvec3 verts[] = {
                glm::dvec3(mesh.verts[pyr[0]]),
                glm::dvec3(mesh.verts[pyr[1]]),
                glm::dvec3(mesh.verts[pyr[2]]),
                glm::dvec3(mesh.verts[pyr[3]]),
                glm::dvec3(mesh.verts[pyr[4]])
            };

            if(_cutType == ECutType::PhysicalPlane)
            {
                if(glm::dot(verts[0], cutNormal) > cutDistance ||
                   glm::dot(verts[1], cutNormal) > cutDistance ||
                   glm::dot(verts[2], cutNormal) > cutDistance ||
                   glm::dot(verts[3], cutNormal) > cutDistance ||
                   glm::dot(verts[4], cutNormal) > cutDistance)
                    continue;
            }


            double quality = pyr.value;
            if(quality >= _qualityCullingMin &&
               quality <= _qualityCullingMax)
            {
                if(_cutType == ECutType::InvertedElements)
                {
                    if(quality >= 0.0)
                        continue;
                    quality = glm::min(-quality, 1.0);
                }
                else
                {
                    quality = glm::max(quality, 0.0);
                }

                for(int f=0; f < MeshPyr::TRI_COUNT; ++f)
                {
                    const MeshTri& tri = MeshPyr::tris[f];
                    glm::dvec3 A = verts[tri[1]] - verts[tri[0]];
                    glm::dvec3 B = verts[tri[2]] - verts[tri[1]];
                    glm::dvec3 normal = glm::normalize(glm::cross(A, B));
                    pushTriangle(faceVertices, normals, qualities,
                                 verts[tri[0]], verts[tri[1]], verts[tri[2]],
                                 normal, quality);
                }

                // Add verts to edge
                for(uint e=0; e < MeshPyr::EDGE_COUNT; ++e)
                {
                    MeshEdge local = MeshPyr::edges[e];
                    MeshEdge global(pyr.v[local.v[0]], pyr.v[local.v[1]]);
                    if(global.v[0] < global.v[1])
                        edgeSet.insert(pair<GLuint, GLuint>(global.v[0], global.v[1]));
                    else
                        edgeSet.insert(pair<GLuint, GLuint>(global.v[1], global.v[0]));
                }
            }
        }
    }


    // Prisms
    if(_priVisibility)
    {
        int priCount = mesh.pris.size();
        for(int i=0; i < priCount; ++i)
        {
            const MeshPri& pri = mesh.pris[i];

            glm::dvec3 verts[] = {
                glm::dvec3(mesh.verts[pri[0]]),
                glm::dvec3(mesh.verts[pri[1]]),
                glm::dvec3(mesh.verts[pri[2]]),
                glm::dvec3(mesh.verts[pri[3]]),
                glm::dvec3(mesh.verts[pri[4]]),
                glm::dvec3(mesh.verts[pri[5]])
            };

            if(_cutType == ECutType::PhysicalPlane)
            {
                if(glm::dot(verts[0], cutNormal) > cutDistance ||
                   glm::dot(verts[1], cutNormal) > cutDistance ||
                   glm::dot(verts[2], cutNormal) > cutDistance ||
                   glm::dot(verts[3], cutNormal) > cutDistance ||
                   glm::dot(verts[4], cutNormal) > cutDistance ||
                   glm::dot(verts[5], cutNormal) > cutDistance)
                    continue;
            }


            double quality = pri.value;
            if(quality >= _qualityCullingMin &&
               quality <= _qualityCullingMax)
            {
                if(_cutType == ECutType::InvertedElements)
                {
                    if(quality >= 0.0)
                        continue;
                    quality = glm::min(-quality, 1.0);
                }
                else
                {
                    quality = glm::max(quality, 0.0);
                }

                for(int f=0; f < MeshPri::TRI_COUNT; ++f)
                {
                    const MeshTri& tri = MeshPri::tris[f];
                    glm::dvec3 A = verts[tri[1]] - verts[tri[0]];
                    glm::dvec3 B = verts[tri[2]] - verts[tri[1]];
                    glm::dvec3 normal = glm::normalize(glm::cross(A, B));
                    pushTriangle(faceVertices, normals, qualities,
                                 verts[tri[0]], verts[tri[1]], verts[tri[2]],
                                 normal, quality);
                }

                // Add verts to edge
                for(uint e=0; e < MeshPri::EDGE_COUNT; ++e)
                {
                    MeshEdge local = MeshPri::edges[e];
                    MeshEdge global(pri.v[local.v[0]], pri.v[local.v[1]]);
                    if(global.v[0] < global.v[1])
                        edgeSet.insert(pair<GLuint, GLuint>(global.v[0], global.v[1]));
                    else
                        edgeSet.insert(pair<GLuint, GLuint>(global.v[1], global.v[0]));
                }
            }
        }
    }


    // Hexahedrons
    if(_hexVisibility)
    {
        int hexCount = mesh.hexs.size();
        for(int i=0; i < hexCount; ++i)
        {
            const MeshHex& hex = mesh.hexs[i];

            glm::dvec3 verts[] = {
                glm::dvec3(mesh.verts[hex[0]]),
                glm::dvec3(mesh.verts[hex[1]]),
                glm::dvec3(mesh.verts[hex[2]]),
                glm::dvec3(mesh.verts[hex[3]]),
                glm::dvec3(mesh.verts[hex[4]]),
                glm::dvec3(mesh.verts[hex[5]]),
                glm::dvec3(mesh.verts[hex[6]]),
                glm::dvec3(mesh.verts[hex[7]])
            };

            if(_cutType == ECutType::PhysicalPlane)
            {
                if(glm::dot(verts[0], cutNormal) > cutDistance ||
                   glm::dot(verts[1], cutNormal) > cutDistance ||
                   glm::dot(verts[2], cutNormal) > cutDistance ||
                   glm::dot(verts[3], cutNormal) > cutDistance ||
                   glm::dot(verts[4], cutNormal) > cutDistance ||
                   glm::dot(verts[5], cutNormal) > cutDistance ||
                   glm::dot(verts[6], cutNormal) > cutDistance ||
                   glm::dot(verts[7], cutNormal) > cutDistance)
                    continue;
            }


            double quality = hex.value;
            if(quality >= _qualityCullingMin &&
               quality <= _qualityCullingMax)
            {
                if(_cutType == ECutType::InvertedElements)
                {
                    if(quality >= 0.0)
                        continue;
                    quality = glm::min(-quality, 1.0);
                }
                else
                {
                    quality = glm::max(quality, 0.0);
                }

                for(int f=0; f < MeshHex::TRI_COUNT; ++f)
                {
                    const MeshTri& tri = MeshHex::tris[f];
                    glm::dvec3 A = verts[tri[1]] - verts[tri[0]];
                    glm::dvec3 B = verts[tri[2]] - verts[tri[1]];
                    glm::dvec3 normal = glm::normalize(glm::cross(A, B));
                    pushTriangle(faceVertices, normals, qualities,
                                 verts[tri[0]], verts[tri[1]], verts[tri[2]],
                                 normal, quality);
                }

                // Add verts to edge
                for(uint e=0; e < MeshHex::EDGE_COUNT; ++e)
                {
                    MeshEdge local = MeshHex::edges[e];
                    MeshEdge global(hex.v[local.v[0]], hex.v[local.v[1]]);
                    if(global.v[0] < global.v[1])
                        edgeSet.insert(pair<GLuint, GLuint>(global.v[0], global.v[1]));
                    else
                        edgeSet.insert(pair<GLuint, GLuint>(global.v[1], global.v[0]));
                }
            }
        }
    }


    // Edge vertices
    edgeVertices.resize(vertCount);
    for(int i=0; i < vertCount; ++i)
        edgeVertices[i] = mesh.verts[i].p;


    // Edge indices
    for(const pair<int, int>& e : edgeSet)
    {
        edgeIndices.push_back(e.first);
        edgeIndices.push_back(e.second);
    }
}

void SurfacicRenderer::pushTriangle(
        std::vector<glm::vec3>& vertices,
        std::vector<signed char>& normals,
        std::vector<unsigned char>& qualities,
        const glm::dvec3& A,
        const glm::dvec3& B,
        const glm::dvec3& C,
        const glm::dvec3& n,
        double quality) const
{

    vertices.push_back(glm::vec3(A));
    vertices.push_back(glm::vec3(B));
    vertices.push_back(glm::vec3(C));

    signed char nx = n.x * 127;
    signed char ny = n.y * 127;
    signed char nz = n.z * 127;
    normals.push_back(nx);
    normals.push_back(ny);
    normals.push_back(nz);
    normals.push_back(nx);
    normals.push_back(ny);
    normals.push_back(nz);
    normals.push_back(nx);
    normals.push_back(ny);
    normals.push_back(nz);

    qualities.push_back(quality * 255);
    qualities.push_back(quality * 255);
    qualities.push_back(quality * 255);
}

void SurfacicRenderer::clearResources()
{
    // Delete old mesh buffers
    glDeleteVertexArrays(1, &_faceVao);
    _faceVao = 0;

    glDeleteBuffers(1, &_faceVbo);
    _faceVbo = 0;

    glDeleteBuffers(1, &_faceNbo);
    _faceNbo = 0;

    glDeleteBuffers(1, &_faceQbo);
    _faceQbo = 0;


    // Delete old edge buffers
    glDeleteBuffers(1, &_edgeVbo);
    _edgeVbo = 0;

    glDeleteBuffers(1, &_edgeIbo);
    _edgeIbo = 0;


    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteRenderbuffers(1, &_shadowDpt);
    _shadowDpt = 0;

    glDeleteTextures(1, &_shadowTex);
    _shadowTex = 0;

    glDeleteFramebuffers(1, &_shadowFbo);
    _shadowFbo = 0;


    glDeleteRenderbuffers(1, &_bloomDpt);
    _bloomDpt = 0;

    glDeleteTextures(1, &_bloomBaseTex);
    _bloomBaseTex = 0;

    glDeleteTextures(1, &_bloomBlurTex);
    _bloomBlurTex = 0;

    glDeleteFramebuffers(1, &_bloomFbo);
    _bloomFbo = 0;


    _buffFaceElemCount = 0;


    _litShader.reset();
    _unlitShader.reset();
    _edgeShader.reset();
    _shadowShader.reset();
    _bloomBlurShader.reset();
    _bloomBlendShader.reset();
    _screenShader.reset();
    _brushShader.reset();
    _grainShader.reset();
}

void SurfacicRenderer::resetResources()
{
    // Generate new mesh buffers
    glGenVertexArrays(1, &_faceVao);
    glBindVertexArray(_faceVao);

    glGenBuffers(1, &_faceVbo);
    glBindBuffer(GL_ARRAY_BUFFER, _faceVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &_faceNbo);
    glBindBuffer(GL_ARRAY_BUFFER, _faceNbo);
    glVertexAttribPointer(1, 3, GL_BYTE, GL_TRUE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &_faceQbo);
    glBindBuffer(GL_ARRAY_BUFFER, _faceQbo);
    glVertexAttribPointer(2, 1, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);


    // Generate new edge buffers
    glGenVertexArrays(1, &_edgeVao);
    glBindVertexArray(_edgeVao);

    glGenBuffers(1, &_edgeVbo);
    glBindBuffer(GL_ARRAY_BUFFER, _edgeVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &_edgeIbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _edgeIbo);
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
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glGenFramebuffers(1, &_shadowFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _shadowFbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _shadowDpt);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _shadowTex, 0);

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    // Bloom
    glGenRenderbuffers(1, &_bloomDpt);
    glBindRenderbuffer(GL_RENDERBUFFER, _bloomDpt);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, 1, 1);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);


    glGenTextures(1, &_bloomBaseTex);
    glBindTexture(GL_TEXTURE_2D, _bloomBaseTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 1, 1,
                 0, GL_RGB, GL_UNSIGNED_INT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &_bloomBlurTex);
    glBindTexture(GL_TEXTURE_2D, _bloomBlurTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 1, 1,
                 0, GL_RGB, GL_UNSIGNED_INT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &_bloomFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _bloomFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _bloomBaseTex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _bloomDpt);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SurfacicRenderer::setupShaders()
{
    // Compile shaders
    _shadowShader.addShader(GL_VERTEX_SHADER, ":/glsl/vertex/Shadow.vert");
    _shadowShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/fragment/Shadow.frag");
    _shadowShader.link();
    _shadowShader.pushProgram();
    _shadowShader.popProgram();

    _litShader.addShader(GL_VERTEX_SHADER, ":/glsl/vertex/LitMesh.vert");
    _litShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/fragment/LitMesh.frag");
    _litShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/generic/QualityLut.glsl");
    _litShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/generic/Lighting.glsl");
    _litShader.link();
    _litShader.pushProgram();
    _litShader.setInt("DepthTex", 0);
    _litShader.popProgram();

    _unlitShader.addShader(GL_VERTEX_SHADER, ":/glsl/vertex/UnlitMesh.vert");
    _unlitShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/fragment/UnlitMesh.frag");
    _unlitShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/generic/QualityLut.glsl");
    _unlitShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/generic/Lighting.glsl");
    _unlitShader.link();
    _unlitShader.pushProgram();
    _unlitShader.popProgram();

    _edgeShader.addShader(GL_VERTEX_SHADER, ":/glsl/vertex/BoldEdge.vert");
    _edgeShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/fragment/BoldEdge.frag");
    _edgeShader.link();
    _edgeShader.pushProgram();
    _edgeShader.popProgram();

    _bloomBlurShader.addShader(GL_VERTEX_SHADER, ":/glsl/vertex/Bloom.vert");
    _bloomBlurShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/fragment/BloomBlur.frag");
    _bloomBlurShader.link();
    _bloomBlurShader.pushProgram();
    _bloomBlurShader.setInt("BloomBase", 2);
    _bloomBlurShader.popProgram();

    _bloomBlendShader.addShader(GL_VERTEX_SHADER, ":/glsl/vertex/Bloom.vert");
    _bloomBlendShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/fragment/BloomBlend.frag");
    _bloomBlendShader.link();
    _bloomBlendShader.pushProgram();
    _bloomBlendShader.setInt("BloomBase", 2);
    _bloomBlendShader.setInt("BloomBlur", 3);
    _bloomBlendShader.popProgram();

    _screenShader.addShader(GL_VERTEX_SHADER, ":/glsl/vertex/Filter.vert");
    _screenShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/fragment/Screen.frag");
    _screenShader.link();
    _screenShader.pushProgram();
    _screenShader.setInt("Base", 2);
    _screenShader.setInt("Filter", 1);
    _screenShader.setVec2f("TexScale", glm::vec2(1.0f));
    _screenShader.popProgram();

    _brushShader.addShader(GL_VERTEX_SHADER, ":/glsl/vertex/Filter.vert");
    _brushShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/fragment/Brush.frag");
    _brushShader.link();
    _brushShader.pushProgram();
    _brushShader.setInt("Base", 3);
    _brushShader.setInt("Filter", 1);
    _brushShader.setVec2f("TexScale", glm::vec2(1.0f));
    _brushShader.popProgram();

    _grainShader.addShader(GL_VERTEX_SHADER, ":/glsl/vertex/Filter.vert");
    _grainShader.addShader(GL_FRAGMENT_SHADER, ":/glsl/fragment/Grain.frag");
    _grainShader.link();
    _grainShader.pushProgram();
    _grainShader.setInt("Base", 2);
    _grainShader.setInt("Filter", 1);
    _grainShader.setVec2f("TexScale", glm::vec2(1.0f));
    _grainShader.popProgram();


    // Set shadow projection view matrix
    _shadowProj = glm::ortho(
            -2.0f, 2.0f,
            -2.0f, 2.0f,
            -2.0f, 2.0f);
}

void SurfacicRenderer::render()
{
    if(_lightingEnabled)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, _bloomFbo);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             _bloomBaseTex,  0);
        glClear(GL_DEPTH_BUFFER_BIT);
    }

    // Render background
    drawBackdrop();


    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, _bloomBlurTex);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, _bloomBaseTex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, filterTex());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _shadowTex);

    glBindVertexArray(_faceVao);


    // If we are rendring inverted elements, switch triangle windings
    if(_cutType == ECutType::InvertedElements)
        glFrontFace(GL_CW);

    // Render shadow map
    if(_lightingEnabled)
    {
        if(_updateShadow)
        {
            int viewport[4];
            glGetIntegerv(GL_VIEWPORT, viewport);

            glBindFramebuffer(GL_FRAMEBUFFER, _shadowFbo);
            glViewport(0, 0, _shadowSize.x, _shadowSize.y);

            glClearColor(1.0, 1.0, 1.0, 1.0);
            glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

            _shadowShader.pushProgram();
            _shadowShader.setVec4f("CutPlaneEq", _virtualCutPlane);
            glDisableVertexAttribArray(1);
            glDisableVertexAttribArray(2);
            glDrawArrays(GL_TRIANGLES, 0, _buffFaceElemCount);
            glEnableVertexAttribArray(1);
            glEnableVertexAttribArray(2);
            _shadowShader.popProgram();

            glBindFramebuffer(GL_FRAMEBUFFER, _bloomFbo);
            glViewport(viewport[0], viewport[1],
                       viewport[2], viewport[3]);

            glGenerateMipmap(GL_TEXTURE_2D);
            _updateShadow = false;
        }

        _litShader.pushProgram();
        _litShader.setVec4f("CutPlaneEq", _virtualCutPlane);
    }
    else
    {
        _unlitShader.pushProgram();
        _unlitShader.setVec4f("CutPlaneEq", _virtualCutPlane);
    }



    // Render mesh
    glBindVertexArray(_faceVao);
    glDrawArrays(GL_TRIANGLES, 0, _buffFaceElemCount);
    GlProgram::popProgram();

    // Render edges
    _edgeShader.pushProgram();
    _edgeShader.setVec4f("CutPlaneEq", _virtualCutPlane);
    glLineWidth(2.5f);
    glBindVertexArray(_edgeVao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _edgeIbo);
    glDrawElements(GL_LINES, _buffEdgeIdxCount, GL_UNSIGNED_INT, 0);
    glLineWidth(1.0f);
    _edgeShader.popProgram();

    // If we were rendring inverted elements, revert windings to default order
    if(_cutType == ECutType::InvertedElements)
        glFrontFace(GL_CCW);


    if(_lightingEnabled)
    {
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);


        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             _bloomBlurTex,  0);
        _bloomBlurShader.pushProgram();
        fullScreenDraw();
        _bloomBlurShader.popProgram();


        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             _bloomBaseTex,  0);
        _bloomBlendShader.pushProgram();
        fullScreenDraw();
        _bloomBlendShader.popProgram();


        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             _bloomBlurTex,  0);
        _screenShader.pushProgram();
        fullScreenDraw();
        _screenShader.popProgram();


        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             _bloomBaseTex,  0);
        _brushShader.pushProgram();
        fullScreenDraw();
        _brushShader.popProgram();


        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        _grainShader.pushProgram();
        fullScreenDraw();
        _grainShader.popProgram();


        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
    }

    glBindVertexArray(0);
}

void SurfacicRenderer::useDiffuseShading()
{
    _lightingEnabled = false;
    _updateShadow = true;
}

void SurfacicRenderer::useSpecularShading()
{
    _lightingEnabled = true;
    _updateShadow = true;
}
