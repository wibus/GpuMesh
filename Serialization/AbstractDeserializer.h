#ifndef GPUMESH_ABSTRACTDESERIALIZER
#define GPUMESH_ABSTRACTDESERIALIZER

#include <memory>
#include <string>

#include <GLM/glm.hpp>

#include "DataStructures/OptionMap.h"

class Mesh;
class AbstractBoundary;
class AbstractSampler;

typedef glm::dmat3 MeshMetric;


class AbstractDeserializer
{
protected:
    AbstractDeserializer();

public:
    virtual ~AbstractDeserializer();

    virtual bool deserialize(
            const std::string& fileName,
            Mesh& mesh,
            std::vector<MeshMetric>& metrics) const = 0;

protected:
    std::shared_ptr<AbstractBoundary> boundary(const std::string& name) const;

private:
    OptionMap<std::shared_ptr<AbstractBoundary>> _boundaries;
};

#endif // GPUMESH_ABSTRACTDESERIALIZER
