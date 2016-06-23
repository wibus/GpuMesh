#ifndef GPUMESH_DEBUGMESHER
#define GPUMESH_DEBUGMESHER

#include <memory>

#include "AbstractMesher.h"


class BoxBoundary;
class TetBoundary;


class DebugMesher : public AbstractMesher
{
public:
    DebugMesher();
    virtual ~DebugMesher();


protected:
    virtual void genSingles(Mesh& mesh, size_t vertexCount);
    virtual void genSquish(Mesh& mesh, size_t vertexCount);
    virtual void genHexGrid(Mesh& mesh, size_t vertexCount);
    virtual void genTetGrid(Mesh& mesh, size_t vertexCount);
    virtual void genCube(Mesh& mesh, size_t vertexCount);
    virtual void genTet(Mesh& mesh, size_t vertexCount);

private:
    std::shared_ptr<BoxBoundary> _boxBoundary;
    std::shared_ptr<TetBoundary> _tetBoundary;
};

#endif // GPUMESH_DEBUGMESHER
