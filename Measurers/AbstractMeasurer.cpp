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
