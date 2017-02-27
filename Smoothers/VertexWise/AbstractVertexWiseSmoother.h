#ifndef GPUMESH_ABSTRACTVERTEXWISESMOOTHER
#define GPUMESH_ABSTRACTVERTEXWISESMOOTHER

#include "../AbstractSmoother.h"

#include "DataStructures/NodeGroups.h"


class AbstractVertexWiseSmoother : public AbstractSmoother
{
protected:
    typedef void (*installCudaFct)(void);
    typedef void (*launchCudaKernelFct)(const NodeGroups::GpuDispatch&);
    AbstractVertexWiseSmoother(
            const std::vector<std::string>& smoothShaders,
            const installCudaFct& installCuda,
            const launchCudaKernelFct& launchCudaKernel);

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

    virtual NodeGroups::GpuDispatcher glslDispatcher() const;

    virtual NodeGroups::GpuDispatcher cudaDispatcher() const;


private:
    bool _initialized;
    std::string _samplingShader;
    std::string _measureShader;
    std::string _evaluationShader;
    std::vector<std::string> _smoothShaders;
    cellar::GlProgram _vertSmoothProgram;

    installCudaFct _installCudaSmoother;
    launchCudaKernelFct _launchCudaKernel;
};

#endif // GPUMESH_ABSTRACTVERTEXWISESMOOTHER
