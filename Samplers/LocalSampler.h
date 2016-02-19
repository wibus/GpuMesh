#ifndef GPUMESH_LOCALSAMPLER
#define GPUMESH_LOCALSAMPLER

#include "AbstractSampler.h"


struct LocalTet
{
    inline LocalTet()
        { v[0] = -1; v[1] = -1; v[2] = -1; v[3] = -1;
          n[0] = -1; n[1] = -1; n[2] = -1; n[3] = -1;}

    inline LocalTet(uint v0, uint v1, uint v2, uint v3)
        { v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
          n[0] = -1; n[1] = -1; n[2] = -1; n[3] = -1;}

    inline LocalTet(const MeshTet& t);

    // Vertices of the tetrahedron
    uint v[4];

    // Neighbors of the tetrahedron
    //   n[0] is the neighbor tetrahedron
    //   at the oposite face of vertex v[0]
    uint n[4];
};


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
            const glm::dvec3& position) const override;


    virtual void releaseDebugMesh() override;
    virtual const Mesh& debugMesh() override;


protected:
    static void tetrahedrizeMesh(
            std::vector<LocalTet>& tets,
            const Mesh& mesh);


private:
    std::shared_ptr<Mesh> _debugMesh;
    std::vector<Metric> _vertMetrics;
    std::vector<LocalTet> _localTets;
    std::vector<uint> _indexCache;
};

#endif // GPUMESH_LOCALSAMPLER
