#ifndef GPUMESH_CPUPARAMETRICMESHER
#define GPUMESH_CPUPARAMETRICMESHER

#include <map>
#include <memory>
#include <functional>

#include "AbstractMesher.h"


class CpuParametricMesher : public AbstractMesher
{
public:
    CpuParametricMesher();
    virtual ~CpuParametricMesher();

    virtual std::vector<std::string> availableMeshModels() const override;

    virtual void generateMesh(
            Mesh& mesh,
            const std::string& modelName,
            size_t vertexCount) override;


protected:
    virtual void genElbowPipe(Mesh& mesh, size_t vertexCount);

    virtual void genStraightPipe(
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

    virtual void genArcPipe(
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

    virtual void insertStackVertices(
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

    std::unique_ptr<MeshBound> _pipeSurface;
    std::unique_ptr<MeshBound> _pipeExtFace;
    std::unique_ptr<MeshBound> _pipeExtEdge;


private:
    // Models
    typedef std::function<void(Mesh&, size_t)> ModelFunc;
    std::map<std::string, ModelFunc> _modelFuncs;
};

#endif // GPUMESH_CPUPARAMETRICMESHER
