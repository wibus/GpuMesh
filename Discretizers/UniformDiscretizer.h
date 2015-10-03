#ifndef GPUMESH_UNIFORMDISCRETIZER
#define GPUMESH_UNIFORMDISCRETIZER

#include "AbstractDiscretizer.h"


class UniformDiscretizer : public AbstractDiscretizer
{
public:
    UniformDiscretizer();
    virtual ~UniformDiscretizer();

    virtual std::shared_ptr<Mesh> gridMesh() const override;

    virtual void discretize(
            const Mesh& mesh,
            const glm::ivec3& gridSize) override;

protected:
    glm::ivec3 cellId(
            const glm::ivec3& gridSize,
            const glm::dvec3& minBounds,
            const glm::dvec3& extents,
            const glm::dvec3& vertPos) const;

private:
    std::shared_ptr<Mesh> _gridMesh;
};

#endif // GPUMESH_UNIFORMDISCRETIZER
