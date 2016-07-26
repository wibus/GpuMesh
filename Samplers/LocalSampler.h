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


    virtual void updateGlslData(const Mesh& mesh) const override;

    virtual void updateCudaData(const Mesh& mesh) const override;

    virtual void clearGlslMemory(const Mesh& mesh) const override;

    virtual void clearCudaMemory(const Mesh& mesh) const override;


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

    mutable GLuint _localTetsSsbo;
    mutable GLuint _refVertsSsbo;
    mutable GLuint _refMetricsSsbo;

    // Debug structures
    mutable int _maxSearchDepth;
    std::vector<Triangle> _surfTris;
    mutable std::vector<glm::dvec4> _failedSamples;
};

#endif // GPUMESH_LOCALSAMPLER
