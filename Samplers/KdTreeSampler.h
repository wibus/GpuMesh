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
    void build(
            KdNode* node,
            int height,
            const Mesh& mesh,
            const glm::dvec3& minBox,
            const glm::dvec3& maxBox,
            std::vector<unsigned int>& xSort,
            std::vector<unsigned int>& ySort,
            std::vector<unsigned int>& zSort,
            std::vector<MeshTet>& tets);

    void buildGpuBuffers(
            KdNode* node,
            std::vector<GpuKdNode>& kdNodes,
            std::vector<GpuTet>& kdTets);

    void meshTree(KdNode* node, Mesh& mesh);

    std::unique_ptr<KdNode> _rootNode;
    std::shared_ptr<Mesh> _debugMesh;
    std::vector<MeshVert> _refVerts;
    std::vector<Metric> _refMetrics;

    GLuint _kdTetsSsbo;
    GLuint _kdNodesSsbo;
    GLuint _refVertsSsbo;
    GLuint _refMetricsSsbo;
    GLuint _metricAtSub;
};

#endif // GPUMESH_KDTREESAMPLER
