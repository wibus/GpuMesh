#ifndef GPUMESH_CPUPARAMETRICMESHER
#define GPUMESH_CPUPARAMETRICMESHER

#include <memory>

#include "AbstractMesher.h"


class CpuParametricMesher : public AbstractMesher
{
public:
    CpuParametricMesher(Mesh& mesh, unsigned int vertCount);
    virtual ~CpuParametricMesher();

    virtual void triangulateDomain() override;


protected:
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

    std::unique_ptr<MeshBound> _pipeSurface;
    std::unique_ptr<MeshBound> _pipeExtFace;
    std::unique_ptr<MeshBound> _pipeExtEdge;
};

#endif // GPUMESH_CPUPARAMETRICMESHER
