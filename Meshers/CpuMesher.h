#ifndef GPUMESH_CPUMESHER
#define GPUMESH_CPUMESHER

#include "AbstractMesher.h"


class CpuMesher : public AbstractMesher
{
public:
    CpuMesher(Mesh& mesh, unsigned int vertCount);
    virtual ~CpuMesher();


protected:
    virtual void smoothMesh() override;


private:

};

#endif // GPUMESH_CPUMESHER
