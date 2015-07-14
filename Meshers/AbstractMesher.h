#ifndef GPUMESH_ABSTRACTMESHER
#define GPUMESH_ABSTRACTMESHER

#include <map>
#include <string>
#include <functional>

#include "DataStructures/Mesh.h"


class AbstractMesher
{
public:
    AbstractMesher();
    virtual ~AbstractMesher() = 0;

    virtual std::vector<std::string> availableMeshModels() const;

    virtual void generateMesh(
            Mesh& mesh,
            const std::string& modelName,
            size_t vertexCount);


protected:
    // Models
    typedef std::function<void(Mesh&, size_t)> ModelFunc;
    std::map<std::string, ModelFunc> _modelFuncs;
};

#endif // GPUMESH_ABSTRACTMESHER
