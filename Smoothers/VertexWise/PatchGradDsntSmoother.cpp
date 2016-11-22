#include "PatchGradDsntSmoother.h"


// Parameters
const int PGDSecurityCycleCount = 5;
const double PGDLocalSizeToNodeShift = 1.0 / 25.0;


// CUDA Drivers
void installCudaPatchGradDsntSmoother(
        int h_securityCycleCount,
        float h_localSizeToNodeShift);
void installCudaPatchGradDsntSmoother()
{
    installCudaPatchGradDsntSmoother(
                PGDSecurityCycleCount,
                PGDLocalSizeToNodeShift);
}

PatchGradDsntSmoother::PatchGradDsntSmoother() :
    GradientDescentSmoother(
        {":/glsl/compute/Smoothing/VertexWise/PatchGradDsnt.glsl"},
        installCudaPatchGradDsntSmoother)
{

}

PatchGradDsntSmoother::~PatchGradDsntSmoother()
{

}

void PatchGradDsntSmoother::smoothMeshGlsl(
            Mesh& mesh,
            const MeshCrew& crew)
{
    if(verifyMeshForGpuLimitations(mesh))
    {
        AbstractVertexWiseSmoother::smoothMeshGlsl(mesh, crew);
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Mesh won't be touched.", "PatchGradDsntSmoother"));
    }
}

void PatchGradDsntSmoother::smoothMeshCuda(
            Mesh& mesh,
            const MeshCrew& crew)
{
    if(verifyMeshForGpuLimitations(mesh))
    {
        AbstractVertexWiseSmoother::smoothMeshCuda(mesh, crew);
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Mesh won't be touched.", "PatchGradDsntSmoother"));
    }
}

bool PatchGradDsntSmoother::verifyMeshForGpuLimitations(
            const Mesh& mesh) const
{
//    if(!mesh.pris.empty())
//    {
//        getLog().postMessage(new Message('E', false,
//            "This mesh contains prismatic elements. "\
//            "This smoother does not support this type of element yet.",
//            "PatchGradDsntSmoother"));
//        return false;
//    }

//    if(!mesh.hexs.empty())
//    {
//        getLog().postMessage(new Message('E', false,
//            "This mesh contains hexahedral elements. "\
//            "This smoother does not support this type of element yet.",
//            "PatchGradDsntSmoother"));
//        return false;
//    }

//    for(const MeshTopo& topo : mesh.topos)
//    {
//        if(topo.neighborElems.size() > PROPOSITION_COUNT)
//        {
//            getLog().postMessage(new Message('E', false,
//                "Some nodes have too many neighbor elements. "\
//                "Maximum " + std::to_string(PROPOSITION_COUNT) +
//                ". A node with " + std::to_string(topo.neighborElems.size()) + " found.",
//                "PatchGradDsntSmoother"));
//            return false;
//        }
//    }

    return true;
}

void PatchGradDsntSmoother::launchCudaKernel(
            const NodeGroups::GpuDispatch& dispatch)
{
    smoothCudaSpawnVertices(dispatch);
}

void PatchGradDsntSmoother::setVertexProgramUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program)
{
    AbstractVertexWiseSmoother::setVertexProgramUniforms(mesh, program);
    program.setFloat("MoveCoeff", SSMoveCoeff);


    if(_offsetsSsbo == 0)
    {
        glGenBuffers(1, &_offsetsSsbo);
    }

    GLuint offsetsBase = mesh.glBufferBinding(EBufferBinding::SPAWN_OFFSETS_BUFFER_BINDING);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, offsetsBase,  _offsetsSsbo);

    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _offsetsSsbo);
        vector<glm::vec4> gpuOffsets(g_offsets.begin(), g_offsets.end());
        size_t offsetsSize = sizeof(decltype(gpuOffsets.front())) * gpuOffsets.size();
        glBufferData(GL_SHADER_STORAGE_BUFFER, offsetsSize, gpuOffsets.data(), GL_STREAM_COPY);
    }
}

void PatchGradDsntSmoother::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationImpl& plotImpl) const
{
    AbstractVertexWiseSmoother::printOptimisationParameters(mesh, plotImpl);
    plotImpl.addSmoothingProperty("Method Name", "Spawn Search");
}

void PatchGradDsntSmoother::smoothVertices(
        Mesh& mesh,
        const MeshCrew& crew,
        const std::vector<uint>& vIds)
{
    std::vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;

    size_t vIdCount = vIds.size();
    for(int v = 0; v < vIdCount; ++v)
    {
        uint vId = vIds[v];

        glm::dvec3& pos = verts[vId].p;

        // Compute local element size
        double localSize = crew.measurer().computeLocalElementSize(mesh, vId);
        double scale = SSMoveCoeff * localSize;


        for(int iter=0; iter < 2; ++iter)
        {
            // Define propositions for new vertex's position
            glm::dvec3 propositions[PROPOSITION_COUNT];
            for(int p=0; p < PROPOSITION_COUNT; ++p)
            {
                propositions[p] = pos + glm::dvec3(g_offsets[p]) * scale;
            }

            const MeshTopo& topo = topos[vId];
            if(topo.snapToBoundary->isConstrained())
            {
                for(uint p=0; p < PROPOSITION_COUNT; ++p)
                    propositions[p] = (*topo.snapToBoundary)(propositions[p]);
            }


            // Choose best position
            uint bestProposition = 0;
            double bestQualityMean = -numeric_limits<double>::infinity();
            for(uint p=0; p < PROPOSITION_COUNT; ++p)
            {
                // Since 'pos' is a reference on vertex's position
                // modifing its value here should be seen by the evaluator
                pos = propositions[p];

                // Compute patch quality
                double patchQuality =
                    crew.evaluator().patchQuality(
                        mesh, crew.sampler(), crew.measurer(), vId);

                if(patchQuality > bestQualityMean)
                {
                    bestQualityMean = patchQuality;
                    bestProposition = p;
                }
            }


            // Update vertex's position
            pos = propositions[bestProposition];

            scale /= 3.0;
        }
    }
}

std::string PatchGradDsntSmoother::glslLauncher() const
{
    return "";
}

glm::ivec3 PatchGradDsntSmoother::layoutWorkgroups(
        const NodeGroups::GpuDispatch& dispatch) const
{
    return glm::ivec3(dispatch.gpuBufferSize, 1, 1);
}
