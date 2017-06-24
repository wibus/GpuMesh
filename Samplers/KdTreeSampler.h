#ifndef GPUMESH_KDTREESAMPLER
#define GPUMESH_KDTREESAMPLER

#include <vector>

#include <GL3/gl3w.h>

#include "AbstractSampler.h"

struct KdNode;
struct MeshVert;
struct GpuKdNode;
struct GpuTet;


class KdTreeSampler : public AbstractSampler
{
public:
    KdTreeSampler();
    virtual ~KdTreeSampler();


    virtual bool isMetricWise() const override;

    virtual bool useComputedMetric() const override;


    virtual void updateGlslData(const Mesh& mesh) const override;

    virtual void updateCudaData(const Mesh& mesh) const override;

    virtual void clearGlslMemory(const Mesh& mesh) const override;

    virtual void clearCudaMemory(const Mesh& mesh) const override;



    virtual void updateAnalyticalMetric(
            const Mesh& mesh) override;

    virtual void updateComputedMetric(
            const Mesh& mesh,
            const std::shared_ptr<LocalSampler>& sampler) override;


    virtual MeshMetric metricAt(
            const glm::dvec3& position,
            uint& cachedRefTet) const override;


    virtual void releaseDebugMesh() override;
    virtual const Mesh& debugMesh() override;


private:
    void build(
            KdNode* node,
            int height,
            const Mesh& mesh,
            const AbstractSampler& localSampler,
            const glm::dvec3& minBox,
            const glm::dvec3& maxBox,
            std::vector<unsigned int>& xSort,
            std::vector<unsigned int>& ySort,
            std::vector<unsigned int>& zSort);

    void buildGpuBuffers(KdNode* node,
            std::vector<GpuKdNode>& kdNodes) const;

    void meshTree(KdNode* node, Mesh& mesh);

    std::unique_ptr<KdNode> _rootNode;
    std::shared_ptr<Mesh> _debugMesh;

    mutable GLuint _kdNodesSsbo;
};

#endif // GPUMESH_KDTREESAMPLER
