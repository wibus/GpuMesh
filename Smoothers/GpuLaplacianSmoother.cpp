#include "GpuLaplacianSmoother.h"

#include <iostream>

using namespace std;


GpuLaplacianSmoother::GpuLaplacianSmoother(
        Mesh &mesh,
        double moveFactor,
        double gainThreshold) :
    AbstractSmoother(mesh, moveFactor, gainThreshold)
{

}

GpuLaplacianSmoother::~GpuLaplacianSmoother()
{

}

void GpuLaplacianSmoother::smoothMesh()
{
}
