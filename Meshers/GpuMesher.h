#ifndef GPUMESH_GPUMESHER
#define GPUMESH_GPUMESHER

#include "AbstractMesher.h"


class GpuMesher : public AbstractMesher
{
public:
    GpuMesher(Mesh& mesh, unsigned int vertCount);
    virtual ~GpuMesher();


protected:


private:

};

#endif // GPUMESH_GPUMESHER
