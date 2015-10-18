#ifndef GPUMESH_ABSTRACTDISCRETIZER
#define GPUMESH_ABSTRACTDISCRETIZER

#include <memory>

#include <GLM/glm.hpp>

namespace cellar
{
    class GlProgram;
}

class Mesh;

typedef glm::dmat3 Metric;


class AbstractDiscretizer
{
protected:
    AbstractDiscretizer(const std::string& name,
                        const std::string& shader);

public:
    virtual ~AbstractDiscretizer();


    virtual bool isMetricWise() const = 0;


    // GLSL Plug-in interface
    virtual std::string discretizationShader() const;

    virtual void installPlugIn(
            const Mesh& mesh,
            cellar::GlProgram& program) const = 0;

    virtual void uploadUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const = 0;


    virtual void discretize(
            const Mesh& mesh,
            int density) = 0;

    virtual Metric metric(
            const glm::dvec3& position) const = 0;


    // Debug mesh
    virtual void releaseDebugMesh() = 0;
    virtual const Mesh& debugMesh() = 0;


protected:
    // Give mesh's provided metric
    Metric vertMetric(const Mesh& mesh, uint vId) const;

    // Interpolate the metric given two samples and a mix ratio
    Metric interpolate(const Metric& m1, const Metric& m2, double a) const;

    // Classic bounding box computation
    void boundingBox(const Mesh& mesh,
                     glm::dvec3& minBounds,
                     glm::dvec3& maxBounds) const;

private:
    std::string _discretizationName;
    std::string _discretizationShader;
};

#endif // GPUMESH_ABSTRACTDISCRETIZER
