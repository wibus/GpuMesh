#ifndef GPUMESH_ABSTRACTSERIALIZER
#define GPUMESH_ABSTRACTSERIALIZER

#include <string>

class Mesh;
class MeshCrew;


class AbstractSerializer
{
protected:
    AbstractSerializer();

public:
    virtual ~AbstractSerializer();

    virtual bool serialize(
            const std::string& fileName,
            const MeshCrew& crew,
            const Mesh& mesh) const = 0;
};

#endif // GPUMESH_ABSTRACTSERIALIZER
