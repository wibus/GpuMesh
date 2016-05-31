#ifndef GPUMESH_CPUPARAMETRICMESHER
#define GPUMESH_CPUPARAMETRICMESHER

#include <memory>

#include "AbstractMesher.h"


class PipeBoundary;


class CpuParametricMesher : public AbstractMesher
{
public:
    CpuParametricMesher();
    virtual ~CpuParametricMesher();


protected:
    virtual void genPipe(Mesh& mesh, size_t vertexCount);
    virtual void genBottle(Mesh& mesh, size_t vertexCount);

    virtual void insertStraightRingPipe(
            Mesh& mesh,
            const glm::dvec3& begin,
            const glm::dvec3& end,
            const glm::dvec3& up,
            double pipeRadius,
            int stackCount,
            int sliceCount,
            int layerCount,
            bool first,
            bool last);

    virtual void insertArcRingPipe(
            Mesh& mesh,
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

    virtual void insertRingStackVertices(
            Mesh& mesh,
            const glm::dvec3& center,
            const glm::dvec4& upBase,
            const glm::dvec3& frontU,
            const glm::dmat4& dSlice,
            double dRadius,
            int sliceCount,
            int layerCount,
            bool isBoundary);

    virtual void meshPipe(
            Mesh& mesh,
            int stackCount,
            int sliceCount,
            int layerCount);

private:
    std::shared_ptr<PipeBoundary> _pipeBoundary;
};

#endif // GPUMESH_CPUPARAMETRICMESHER
