#ifndef GPUMESH_UNIFORMDISCRETIZER
#define GPUMESH_UNIFORMDISCRETIZER

#include "AbstractDiscretizer.h"

class UniformGrid;


class UniformDiscretizer : public AbstractDiscretizer
{
public:
    UniformDiscretizer();
    virtual ~UniformDiscretizer();


    virtual void discretize(
            const Mesh& mesh,
            const glm::ivec3& gridSize) override;

    virtual Metric metric(
            const glm::dvec3& position) const override;

    virtual double distance(
            const glm::dvec3& a,
            const glm::dvec3& b) const override;


    virtual void installPlugIn(
            const Mesh& mesh,
            cellar::GlProgram& program) const override;

    virtual void uploadPlugInUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const override;

    virtual void releaseDebugMesh() override;
    virtual std::shared_ptr<Mesh> debugMesh() override;


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

#endif // GPUMESH_UNIFORMDISCRETIZER
