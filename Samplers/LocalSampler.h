#ifndef GPUMESH_LOCALSAMPLER
#define GPUMESH_LOCALSAMPLER

#include <GL3/gl3w.h>

#include "AbstractSampler.h"

class MeshLocalTet;
class Triangle;


class LocalSampler : public AbstractSampler
{
public:
    LocalSampler();
    virtual ~LocalSampler();


    virtual bool isMetricWise() const override;

    virtual void initialize() override;


    virtual void installPlugin(
            const Mesh& mesh,
            cellar::GlProgram& program) const override;

    virtual void setPluginUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const override;


    virtual void setReferenceMesh(
            const Mesh& mesh) override;

    virtual Metric metricAt(
            const glm::dvec3& position,
            uint& cachedRefTet) const override;


    virtual void releaseDebugMesh() override;
    virtual const Mesh& debugMesh() override;


private:
    std::shared_ptr<Mesh> _debugMesh;

    std::vector<MeshVert> _refVerts;
    std::vector<Metric>   _refMetrics;
    std::vector<MeshLocalTet> _localTets;

    GLuint _localTetsSsbo;
    GLuint _refVertsSsbo;
    GLuint _refMetricsSsbo;

    // Debug structures
    mutable int _maxSearchDepth;
    std::vector<Triangle> _surfTris;
    mutable std::vector<glm::dvec4> _failedSamples;
};

#endif // GPUMESH_LOCALSAMPLER
