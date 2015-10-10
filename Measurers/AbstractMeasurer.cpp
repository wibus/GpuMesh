#include "AbstractMeasurer.h"

#include "DataStructures/Mesh.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Discretizers/AbstractDiscretizer.h"

using namespace std;


AbstractMeasurer::AbstractMeasurer(
        const string& name,
        const string& shader) :
    _measureName(name),
    _measureShader(shader),
    _frameworkShader(":/shaders/compute/Measuring/Framework.glsl")
{

}

AbstractMeasurer::~AbstractMeasurer()
{

}

std::string AbstractMeasurer::measureShader() const
{
    return _measureShader;
}

void AbstractMeasurer::installPlugIn(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    program.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        _frameworkShader.c_str()
    });

    program.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        _measureShader.c_str()
    });
}

void AbstractMeasurer::uploadUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{

}

double AbstractMeasurer::computeLocalElementSize(
        const Mesh& mesh,
        size_t vId) const
{
    const std::vector<MeshVert>& verts = mesh.verts;

    const glm::dvec3& pos = verts[vId].p;
    const MeshTopo& topo = mesh.topos[vId];
    const vector<MeshNeigVert>& neigVerts = topo.neighborVerts;

    double totalSize = 0.0;
    size_t neigVertCount = neigVerts.size();
    for(size_t n=0; n < neigVertCount; ++n)
    {
        totalSize += glm::length(pos - verts[neigVerts[n].v].p);
    }

    return totalSize / neigVertCount;
}

glm::dvec3 AbstractMeasurer::computeVertexEquilibrium(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        size_t vId) const
{
    const std::vector<MeshVert>& verts = mesh.verts;
    const std::vector<MeshTet>& tets = mesh.tets;
    const std::vector<MeshPri>& pris = mesh.pris;
    const std::vector<MeshHex>& hexs = mesh.hexs;

    const MeshTopo& topo = mesh.topos[vId];

    uint totalVertCount = 0;
    glm::dvec3 patchCenter(0.0);
    uint neigElemCount = topo.neighborElems.size();
    for(uint n=0; n < neigElemCount; ++n)
    {
        const MeshNeigElem& neigElem = topo.neighborElems[n];
        switch(neigElem.type)
        {
        case MeshTet::ELEMENT_TYPE:
            totalVertCount += MeshTet::VERTEX_COUNT - 1;
            for(uint i=0; i < MeshTet::VERTEX_COUNT; ++i)
                patchCenter += verts[tets[neigElem.id].v[i]].p;
            break;

        case MeshPri::ELEMENT_TYPE:
            totalVertCount += MeshPri::VERTEX_COUNT - 1;
            for(uint i=0; i < MeshPri::VERTEX_COUNT; ++i)
                patchCenter += verts[pris[neigElem.id].v[i]].p;
            break;

        case MeshHex::ELEMENT_TYPE:
            totalVertCount += MeshHex::VERTEX_COUNT - 1;
            for(uint i=0; i < MeshHex::VERTEX_COUNT; ++i)
                patchCenter += verts[hexs[neigElem.id].v[i]].p;
            break;
        }
    }

    const glm::dvec3& pos = verts[vId].p;
    patchCenter = (patchCenter - pos * double(neigElemCount))
                    / double(totalVertCount);

    return patchCenter;
}

double AbstractMeasurer::computePatchQuality(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator,
            size_t vId) const
{
    const std::vector<MeshTet>& tets = mesh.tets;
    const std::vector<MeshPri>& pris = mesh.pris;
    const std::vector<MeshHex>& hexs = mesh.hexs;

    const MeshTopo& topo = mesh.topos[vId];

    size_t neigElemCount = topo.neighborElems.size();

    double patchWeight = 0.0;
    double patchQuality = 1.0;
    for(size_t n=0; n < neigElemCount; ++n)
    {
        const MeshNeigElem& neigElem = topo.neighborElems[n];

        switch(neigElem.type)
        {
        case MeshTet::ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                evaluator.tetQuality(mesh, tets[neigElem.id]));
            break;

        case MeshPri::ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                evaluator.priQuality(mesh, pris[neigElem.id]));
            break;

        case MeshHex::ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                evaluator.hexQuality(mesh, hexs[neigElem.id]));
            break;
        }
    }

    return finalizePatchQuality(patchQuality, patchWeight);
}

glm::dvec3 AbstractMeasurer::computeSpringForce(
        const AbstractDiscretizer& discretizer,
        const glm::dvec3& pi,
        const glm::dvec3& pj) const
{
    if(pi == pj)
        return glm::dvec3();

    double d = discretizer.distance(pi, pj);
    glm::dvec3 u = (pi - pj) / d;

    double d2 = d * d;
    double d4 = d2 * d2;

    //double f = (1 - d4) * glm::exp(-d4);
    double f = (1-d2)*glm::exp(-d2/4.0)/2.0;

    return f * u;
}

void AbstractMeasurer::accumulatePatchQuality(
        double& patchQuality,
        double& patchWeight,
        double elemQuality) const
{
    patchQuality = glm::min(
        glm::min(patchQuality, elemQuality),  // If sign(patch) != sign(elem)
        glm::min(patchQuality * elemQuality,  // If sign(patch) & sign(elem) > 0
                 patchQuality + elemQuality));// If sign(patch) & sign(elem) < 0
}

double AbstractMeasurer::finalizePatchQuality(
        double patchQuality,
        double patchWeight) const
{
    return patchQuality;
}
