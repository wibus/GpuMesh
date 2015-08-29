#ifndef GPUMESH_DEBUGMESHER
#define GPUMESH_DEBUGMESHER

#include <memory>

#include "AbstractMesher.h"


class DebugMesher : public AbstractMesher
{
public:
    DebugMesher();
    virtual ~DebugMesher();


protected:
    virtual void genSingles(Mesh& mesh, size_t vertexCount);
    virtual void genSquish(Mesh& mesh, size_t vertexCount);
};

#endif // GPUMESH_DEBUGMESHER
