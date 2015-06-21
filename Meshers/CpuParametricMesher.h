#ifndef GPUMESH_CPUPARAMETRICMESHER
#define GPUMESH_CPUPARAMETRICMESHER

#include "CpuMesher.h"


class CpuParametricMesher : public CpuMesher
{
public:
    CpuParametricMesher(Mesh& mesh, unsigned int vertCount);
    virtual ~CpuParametricMesher();


protected:
    virtual void triangulateDomain() override;
    virtual void genStraightTube(
            const glm::dvec3& begin,
            const glm::dvec3& end,
            const glm::dvec3& up,
            double tubeRadius,
            int stackCount,
            int sliceCount,
            int layerCount);


private:
};

#endif // GPUMESH_CPUPARAMETRICMESHER
