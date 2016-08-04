#ifndef GPUMESH_ABSTRACTVERTEXWISESMOOTHER
#define GPUMESH_ABSTRACTVERTEXWISESMOOTHER

#include "../AbstractSmoother.h"

#include "DataStructures/NodeGroups.h"


class AbstractVertexWiseSmoother : public AbstractSmoother
{
protected:
    AbstractVertexWiseSmoother(
            const std::vector<std::string>& smoothShaders,
            const installCudaFct installCuda);

public:
    ~AbstractVertexWiseSmoother();


    virtual void smoothMeshSerial(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void smoothMeshThread(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void smoothMeshGlsl(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void smoothMeshCuda(
            Mesh& mesh,
            const MeshCrew& crew) override;


protected:    
    virtual void lauchCudaKernel(
            const NodeGroups::GpuDispatch& dispatch);

    virtual void initializeProgram(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program);

    virtual void printOptimisationParameters(
            const Mesh& mesh,
            OptimizationImpl& plotImpl) const override;

    virtual void smoothVertices(
            Mesh& mesh,
            const MeshCrew& crew,
            const std::vector<uint>& vIds) = 0;

    virtual std::string glslLauncher() const;

    virtual glm::ivec3 layoutWorkgroups(
            const NodeGroups::GpuDispatch& dispatch) const;


private:
    static const size_t WORKGROUP_SIZE;

    bool _initialized;
    std::string _samplingShader;
    std::string _measureShader;
    std::string _evaluationShader;
    std::vector<std::string> _smoothShaders;
    cellar::GlProgram _vertSmoothProgram;
};

#endif // GPUMESH_ABSTRACTVERTEXWISESMOOTHER
