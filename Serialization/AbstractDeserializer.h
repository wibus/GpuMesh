#ifndef GPUMESH_ABSTRACTDESERIALIZER
#define GPUMESH_ABSTRACTDESERIALIZER

#include <string>

class Mesh;


class AbstractDeserializer
{
protected:
    AbstractDeserializer();

public:
    virtual ~AbstractDeserializer();

    virtual bool deserialize(
            const std::string& fileName,
            Mesh& mesh) const = 0;
};

#endif // GPUMESH_ABSTRACTDESERIALIZER
