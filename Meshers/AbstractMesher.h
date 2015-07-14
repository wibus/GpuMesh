#ifndef GPUMESH_ABSTRACTMESHER
#define GPUMESH_ABSTRACTMESHER

#include <string>

#include "DataStructures/Mesh.h"


class AbstractMesher
{
public:
    AbstractMesher();
    virtual ~AbstractMesher() = 0;

    virtual std::vector<std::string> availableMeshModels() const = 0;

    virtual void generateMesh(
            Mesh& mesh,
            const std::string& modelName,
            size_t vertexCount) = 0;
};

#endif // GPUMESH_ABSTRACTMESHER
