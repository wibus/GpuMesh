#include "GpuMesher.h"


GpuMesher::GpuMesher(Mesh& mesh, unsigned int vertCount) :
    AbstractMesher(mesh, vertCount)
{
}

GpuMesher::~GpuMesher()
{
}
