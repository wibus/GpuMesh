#ifndef GPUMESH_UNIFORMSAMPLER
#define GPUMESH_UNIFORMSAMPLER

#include "AbstractSampler.h"

class UniformGrid;


class UniformSampler : public AbstractSampler
{
public:
    UniformSampler();
    virtual ~UniformSampler();


    virtual bool isMetricWise() const override;


    virtual void setReferenceMesh(
            const Mesh& mesh,
            int density) override;

    virtual Metric metricAt(
            const glm::dvec3& position,
            uint cacheId) const override;


    virtual void releaseDebugMesh() override;
    virtual const Mesh& debugMesh() override;


protected:
    glm::ivec3 cellId(
            const glm::ivec3& gridSize,
            const glm::dvec3& minBounds,
            const glm::dvec3& extents,
            const glm::dvec3& vertPos) const;

    void meshGrid(UniformGrid& grid, Mesh& mesh);


private:
    std::unique_ptr<UniformGrid> _grid;
    std::shared_ptr<Mesh> _debugMesh;
};

#endif // GPUMESH_UNIFORMSAMPLER
