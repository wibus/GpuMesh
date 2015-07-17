#ifndef GPUMESH_ABSTRACTMESHER
#define GPUMESH_ABSTRACTMESHER

#include <functional>

#include "DataStructures/Mesh.h"
#include "DataStructures/OptionMap.h"


class AbstractMesher
{
public:
    AbstractMesher();
    virtual ~AbstractMesher() = 0;

    virtual OptionMapDetails availableMeshModels() const;

    virtual void generateMesh(
            Mesh& mesh,
            const std::string& modelName,
            size_t vertexCount);


protected:
    // Models
    typedef std::function<void(Mesh&, size_t)> ModelFunc;
    OptionMap<ModelFunc> _modelFuncs;
};

#endif // GPUMESH_ABSTRACTMESHER
