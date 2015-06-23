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

    virtual void genStraightPipe(
            const glm::dvec3& begin,
            const glm::dvec3& end,
            const glm::dvec3& up,
            double pipeRadius,
            int stackCount,
            int sliceCount,
            int layerCount,
            bool first,
            bool last);

    virtual void genArcPipe(
            const glm::dvec3& center,
            const glm::dvec3& rotationAxis,
            const glm::dvec3& dirBegin,
            const glm::dvec3& upBegin,
            double arcAngle,
            double arcRadius,
            double pipeRadius,
            int stackCount,
            int sliceCount,
            int layerCount,
            bool first,
            bool last);

    virtual void insertStackVertices(
            const glm::dvec3& center,
            const glm::dvec4& upBase,
            const glm::dvec3& frontU,
            const glm::dmat4& dSlice,
            double dRadius,
            int sliceCount,
            int layerCount,
            bool isBoundary);

    virtual void meshPipe(
            int stackCount,
            int sliceCount,
            int layerCount);

private:
};

#endif // GPUMESH_CPUPARAMETRICMESHER
