#ifndef GPUMESH_ABSTRACTMESHER
#define GPUMESH_ABSTRACTMESHER

#include <string>

#include "DataStructures/Mesh.h"


class AbstractMesher
{
public:
    AbstractMesher(Mesh& mesh, unsigned int vertCount);
    virtual ~AbstractMesher() = 0;

    virtual void triangulateDomain() = 0;


protected:
    Mesh& _mesh;
    unsigned int _vertCount;
};

#endif // GPUMESH_ABSTRACTMESHER
