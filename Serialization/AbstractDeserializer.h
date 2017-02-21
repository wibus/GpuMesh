#ifndef GPUMESH_ABSTRACTDESERIALIZER
#define GPUMESH_ABSTRACTDESERIALIZER

#include <memory>
#include <string>

#include "DataStructures/OptionMap.h"

class Mesh;
class AbstractBoundary;


class AbstractDeserializer
{
protected:
    AbstractDeserializer();

public:
    virtual ~AbstractDeserializer();

    virtual bool deserialize(
            const std::string& fileName,
            Mesh& mesh) const = 0;

protected:
    std::shared_ptr<AbstractBoundary> boundary(const std::string& name) const;

private:
    OptionMap<std::shared_ptr<AbstractBoundary>> _boundaries;
};

#endif // GPUMESH_ABSTRACTDESERIALIZER
