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


protected:
    glm::ivec3 cellId(
            const UniformGrid& grid,
            const glm::dvec3& vertPos) const;

    void meshGrid(UniformGrid& grid, Mesh& mesh);


private:
    std::unique_ptr<UniformGrid> _grid;
    std::shared_ptr<Mesh> _debugMesh;
};

#endif // GPUMESH_UNIFORMSAMPLER
