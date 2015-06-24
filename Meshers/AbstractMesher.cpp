#include "AbstractMesher.h"


AbstractMesher::AbstractMesher(Mesh& mesh, unsigned int vertCount) :
    _mesh(mesh),
    _vertCount(vertCount)
{

}

AbstractMesher::~AbstractMesher()
{

}
