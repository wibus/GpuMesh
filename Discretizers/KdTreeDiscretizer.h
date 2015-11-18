#ifndef GPUMESH_KDTREEDISCRETIZER
#define GPUMESH_KDTREEDISCRETIZER

#include <vector>

#include <GL3/gl3w.h>

#include "AbstractDiscretizer.h"

struct KdNode;
class MeshVert;
struct GpuKdNode;
struct GpuTet;


class KdTreeDiscretizer : public AbstractDiscretizer
{
public:
    KdTreeDiscretizer();
    virtual ~KdTreeDiscretizer();


    virtual bool isMetricWise() const;


    virtual void installPlugin(
            const Mesh& mesh,
            cellar::GlProgram& program) const override;

    virtual void setPluginUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const override;

    virtual void setupPluginExecution(
            const Mesh& mesh,
            const cellar::GlProgram& program) const override;


    virtual void discretize(
            const Mesh& mesh,
            int density) override;

    virtual Metric metricAt(
            const glm::dvec3& position) const override;


    virtual void releaseDebugMesh() override;
    virtual const Mesh& debugMesh() override;


private:
    void build(
            KdNode* node,
            int height,
            const Mesh& mesh,
            const glm::dvec3& minBox,
            const glm::dvec3& maxBox,
            std::vector<uint>& xSort,
            std::vector<uint>& ySort,
            std::vector<uint>& zSort,
            std::vector<MeshTet>& tets);

    void buildGpuBuffers(
            KdNode* node,
            std::vector<GpuKdNode>& kdNodes,
            std::vector<GpuTet>& kdTets);

    void meshTree(KdNode* node, Mesh& mesh);

    static bool tetParams(const std::vector<MeshVert>& verts, const MeshTet& tet,
                          const glm::dvec3& p, double coor[4]);

    std::unique_ptr<KdNode> _rootNode;
    std::shared_ptr<Mesh> _debugMesh;
    std::vector<Metric> _vertMetrics;

    GLuint _kdNodesSsbo;
    GLuint _kdTetsSsbo;
    GLuint _kdMetricsSsbo;
    GLuint _metricAtSub;
};

#endif // GPUMESH_KDTREEDISCRETIZER
