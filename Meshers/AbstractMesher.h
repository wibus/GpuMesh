#ifndef GPUMESH_ABSTRACTMESHER
#define GPUMESH_ABSTRACTMESHER

#include <string>

#include "DataStructures/Mesh.h"


class AbstractMesher
{
public:
    AbstractMesher(unsigned int vertCount);
    virtual ~AbstractMesher() = 0;

    virtual void triangulateDomain(Mesh& mesh) = 0;


protected:
    unsigned int _vertCount;
};

#endif // GPUMESH_ABSTRACTMESHER
