#ifndef GPUMESH_ABSTRACTDISCRETIZER
#define GPUMESH_ABSTRACTDISCRETIZER

#include <memory>

#include <GLM/glm.hpp>

namespace cellar
{
    class GlProgram;
}

class Mesh;

typedef glm::dvec4 Metric;


class AbstractDiscretizer
{
protected:
    AbstractDiscretizer();

public:
    virtual ~AbstractDiscretizer();


    virtual void discretize(
            const Mesh& mesh,
            const glm::ivec3& gridSize) = 0;

    virtual Metric metricAt(
            const glm::dvec3& position) const = 0;


    virtual void installPlugIn(
            const Mesh& mesh,
            cellar::GlProgram& program) const = 0;

    virtual void uploadPlugInUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const = 0;

    virtual void releaseDebugMesh() = 0;
    virtual std::shared_ptr<Mesh> debugMesh() = 0;


protected:
    Metric vertMetric(const Mesh& mesh, uint vId) const;
    void boundingBox(const Mesh& mesh,
                     glm::dvec3& minBounds,
                     glm::dvec3& maxBounds) const;
};

#endif // GPUMESH_ABSTRACTDISCRETIZER
