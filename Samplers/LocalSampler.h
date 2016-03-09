#ifndef GPUMESH_LOCALSAMPLER
#define GPUMESH_LOCALSAMPLER

#include <GL3/gl3w.h>

#include "AbstractSampler.h"

class LocalTet;


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

    virtual void setupPluginExecution(
            const Mesh& mesh,
            const cellar::GlProgram& program) const override;


    virtual void setReferenceMesh(
            const Mesh& mesh,
            int density) override;

    virtual Metric metricAt(
            const glm::dvec3& position,
            uint cacheId) const override;


    virtual void releaseDebugMesh() override;
    virtual const Mesh& debugMesh() override;


private:
    std::shared_ptr<Mesh> _debugMesh;
    std::vector<Metric>   _refMetrics;
    std::vector<LocalTet> _localTets;
    std::vector<MeshVert> _refVerts;

    GLuint _localTetsSsbo;
    GLuint _localCacheSsbo;
    GLuint _refVertsSsbo;
    GLuint _refMetricsSsbo;
    GLuint _metricAtSub;

    mutable std::vector<uint> _localCache;
};

#endif // GPUMESH_LOCALSAMPLER
