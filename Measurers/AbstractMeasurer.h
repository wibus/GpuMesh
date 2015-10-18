#ifndef GPUMESH_ABSTRACTMEASURER
#define GPUMESH_ABSTRACTMEASURER

#include <string>
#include <vector>

#include <GLM/glm.hpp>

namespace cellar
{
    class GlProgram;
}

class Mesh;
class AbstractEvaluator;
class AbstractDiscretizer;


class AbstractMeasurer
{
protected:
    AbstractMeasurer(const std::string& name,
                     const std::string& shader);

public:
    virtual ~AbstractMeasurer();


    // GLSL Plug-in interface
    virtual std::string measureShader() const;

    virtual void installPlugIn(
            const Mesh& mesh,
            cellar::GlProgram& program) const;

    virtual void uploadUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const;


    virtual double measuredDistance(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3& a,
            const glm::dvec3& b) const = 0;

    virtual double computeLocalElementSize(
            const Mesh& mesh,
            size_t vId) const = 0;

    virtual glm::dvec3 computeVertexEquilibrium(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            size_t vId) const = 0;

    virtual double computePatchQuality(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator,
            size_t vId) const = 0;


protected:
    virtual void accumulatePatchQuality(
            double& patchQuality,
            double& patchWeight,
            double elemQuality) const;

    virtual double finalizePatchQuality(
            double patchQuality,
            double patchWeight) const;

private:
    std::string _measureName;
    std::string _frameworkShader;
    std::string _measureShader;
};

#endif // GPUMESH_ABSTRACTMEASURER
