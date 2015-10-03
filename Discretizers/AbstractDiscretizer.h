#ifndef GPUMESH_ABSTRACTDISCRETIZER
#define GPUMESH_ABSTRACTDISCRETIZER

#include <memory>

#include <GLM/glm.hpp>

class Mesh;


class AbstractDiscretizer
{
protected:
    AbstractDiscretizer();

public:
    virtual ~AbstractDiscretizer();

    virtual std::shared_ptr<Mesh> gridMesh() const = 0;

    virtual void discretize(
            const Mesh& mesh,
            const glm::ivec3& gridSize) = 0;

protected:
    double vertValue(const Mesh& mesh, uint vId) const;
    void boundingBox(const Mesh& mesh,
                     glm::dvec3& minBounds,
                     glm::dvec3& maxBounds) const;
};

#endif // GPUMESH_ABSTRACTDISCRETIZER
