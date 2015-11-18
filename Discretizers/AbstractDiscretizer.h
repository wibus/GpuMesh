#ifndef GPUMESH_ABSTRACTDISCRETIZER
#define GPUMESH_ABSTRACTDISCRETIZER

#include <memory>
#include <vector>

#include <GLM/glm.hpp>

namespace cellar
{
    class GlProgram;
}

class Mesh;
class MeshTet;
class MeshPri;
class MeshHex;

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

    virtual void installPlugin(
            const Mesh& mesh,
            cellar::GlProgram& program) const;

    virtual void setPluginUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const;

    virtual void setupPluginExecution(
            const Mesh& mesh,
            const cellar::GlProgram& program) const;


    virtual void discretize(
            const Mesh& mesh,
            int density) = 0;

    virtual Metric metricAt(
            const glm::dvec3& position) const = 0;


    // Debug mesh
    virtual void releaseDebugMesh() = 0;
    virtual const Mesh& debugMesh() = 0;


protected:
    // Give mesh's provided metric
    Metric vertMetric(const Mesh& mesh, uint vId) const;
    Metric vertMetric(const glm::dvec3& position) const;

    // Interpolate the metric given two samples and a mix ratio
    Metric interpolateMetrics(const Metric& m1, const Metric& m2, double a) const;

    // Classic bounding box computation
    void boundingBox(const Mesh& mesh,
                     glm::dvec3& minBounds,
                     glm::dvec3& maxBounds) const;

    static void tetrahedrizeMesh(const Mesh& mesh, std::vector<MeshTet>& tets);

private:
    std::string _discretizationName;
    std::string _discretizationShader;
    std::string _baseShader;
};

#endif // GPUMESH_ABSTRACTDISCRETIZER
