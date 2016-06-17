#include "NelderMeadSmoother.h"

#include <limits>

#include <CellarWorkbench/Misc/Log.h>

#include "Boundaries/Constraints/AbstractConstraint.h"
#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"

using namespace std;
using namespace cellar;


// Parameters
const double NMValueConvergence = 0.000100;
const int NMSecurityCycleCount = 12;
const double NMLocalSizeToNodeShift = 1.0 / 24.0;
const double NMAlpha = 1.0;
const double NMBeta = 0.5;
const double NMGamma = 2.0;
const double NMDelta = 0.5;

// CUDA Drivers
void installCudaNelderMeadSmoother(
        float h_valueConvergence,
        int h_securityCycleCount,
        float h_localSizeToNodeShift,
        float h_alpha,
        float h_beta,
        float h_gamma,
        float h_delta);
void installCudaNelderMeadSmoother()
{
    installCudaNelderMeadSmoother(
        NMValueConvergence,
        NMSecurityCycleCount,
        NMLocalSizeToNodeShift,
        NMAlpha,
        NMBeta,
        NMGamma,
        NMDelta);
}


NelderMeadSmoother::NelderMeadSmoother() :
    AbstractVertexWiseSmoother(
        {":/glsl/compute/Smoothing/VertexWise/NelderMead.glsl"},
        installCudaNelderMeadSmoother)
{

}

NelderMeadSmoother::~NelderMeadSmoother()
{

}

void NelderMeadSmoother::setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program)
{
    AbstractVertexWiseSmoother::setVertexProgramUniforms(mesh, program);
    program.setFloat("ValueConvergence", NMValueConvergence);
    program.setInt("SecurityCycleCount", NMSecurityCycleCount);
    program.setFloat("LocalSizeToNodeShift", NMLocalSizeToNodeShift);
    program.setFloat("Alpha", NMAlpha);
    program.setFloat("Beta", NMBeta);
    program.setFloat("Gamma", NMGamma);
    program.setFloat("Delta", NMDelta);
}

void NelderMeadSmoother::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{
    AbstractVertexWiseSmoother::printOptimisationParameters(mesh, plot);
    plot.addSmoothingProperty("Method Name", "Local Optimization");
    plot.addSmoothingProperty("Value Convergence", to_string(NMValueConvergence));
    plot.addSmoothingProperty("Security Cycle Count", to_string(NMSecurityCycleCount));
    plot.addSmoothingProperty("Local Size to Node Shift", to_string(NMLocalSizeToNodeShift));
    plot.addSmoothingProperty("Reflexion", to_string(NMAlpha));
    plot.addSmoothingProperty("Contraction", to_string(NMBeta));
    plot.addSmoothingProperty("Expansion", to_string(NMGamma));
    plot.addSmoothingProperty("Shrinkage", to_string(NMDelta));
}

void NelderMeadSmoother::smoothVertices(
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


        // Compute local element size
        double localSize =
            crew.measurer().computeLocalElementSize(mesh, vId);


        // Initialize node shift distance
        double nodeShift = localSize * NMLocalSizeToNodeShift;


        glm::dvec3& pos = verts[vId].p;
        const MeshTopo& topo = topos[vId];
        glm::dvec4 vo(pos, crew.evaluator().patchQuality(
            mesh, crew.sampler(), crew.measurer(), vId));

        glm::dvec4 simplex[MeshTet::VERTEX_COUNT] = {
            glm::dvec4(pos + glm::dvec3(nodeShift, 0, 0), 0),
            glm::dvec4(pos + glm::dvec3(0, nodeShift, 0), 0),
            glm::dvec4(pos + glm::dvec3(0, 0, nodeShift), 0),
            vo
        };

        int cycle = 0;
        bool reset = false;
        bool terminated = false;
        while(!terminated)
        {
            for(uint p=0; p < MeshTet::VERTEX_COUNT-1; ++p)
            {
                // Since 'pos' is a reference on vertex's position
                // modifing its value here should be seen by the evaluator
                pos = glm::dvec3(simplex[p]);
                if(topo.snapToBoundary->isConstrained())
                    pos = (*topo.snapToBoundary)(pos);


                // Compute patch quality
                simplex[p] = glm::dvec4(pos,
                    crew.evaluator().patchQuality(
                        mesh, crew.sampler(), crew.measurer(), vId));
            }

            // Mini bubble sort
            if(simplex[0].w > simplex[1].w)
                std::swap(simplex[0], simplex[1]);
            if(simplex[1].w > simplex[2].w)
                std::swap(simplex[1], simplex[2]);
            if(simplex[2].w > simplex[3].w)
                std::swap(simplex[2], simplex[3]);
            if(simplex[0].w > simplex[1].w)
                std::swap(simplex[0], simplex[1]);
            if(simplex[1].w > simplex[2].w)
                std::swap(simplex[1], simplex[2]);
            if(simplex[0].w > simplex[1].w)
                std::swap(simplex[0], simplex[1]);


            for(; cycle < NMSecurityCycleCount; ++cycle)
            {
                // Centroid
                glm::dvec3 c = 1/3.0 * (
                    glm::dvec3(simplex[1]) +
                    glm::dvec3(simplex[2]) +
                    glm::dvec3(simplex[3]));

                double f = 0.0;

                // Reflect
                pos = c + NMAlpha*(c - glm::dvec3(simplex[0]));
                if(topo.snapToBoundary->isConstrained()) pos = (*topo.snapToBoundary)(pos);
                double fr = f = crew.evaluator().patchQuality(
                    mesh, crew.sampler(), crew.measurer(), vId);

                glm::dvec3 xr = pos;

                // Expand
                if(simplex[3].w < fr)
                {
                    pos = c + NMGamma*(pos - c);
                    if(topo.snapToBoundary->isConstrained()) pos = (*topo.snapToBoundary)(pos);
                    double fe = f = crew.evaluator().patchQuality(
                        mesh, crew.sampler(), crew.measurer(), vId);

                    if(fe <= fr)
                    {
                        pos = xr;
                        f = fr;
                    }
                }
                // Contract
                else if(simplex[1].w >= fr)
                {
                    // Outside
                    if(fr > simplex[0].w)
                    {
                        pos = c + NMBeta*(xr - c);
                        if(topo.snapToBoundary->isConstrained()) pos = (*topo.snapToBoundary)(pos);
                        f = crew.evaluator().patchQuality(
                            mesh, crew.sampler(), crew.measurer(), vId);
                    }
                    // Inside
                    else
                    {
                        pos = c + NMBeta*(glm::dvec3(simplex[0]) - c);
                        if(topo.snapToBoundary->isConstrained()) pos = (*topo.snapToBoundary)(pos);
                        f = crew.evaluator().patchQuality(
                            mesh, crew.sampler(), crew.measurer(), vId);
                    }
                }

                // Insert new vertex in the working simplex
                glm::dvec4 vertex(pos, f);
                if(vertex.w > simplex[3].w)
                    std::swap(simplex[3], vertex);
                if(vertex.w > simplex[2].w)
                    std::swap(simplex[2], vertex);
                if(vertex.w > simplex[1].w)
                    std::swap(simplex[1], vertex);
                if(vertex.w > simplex[0].w)
                    std::swap(simplex[0], vertex);


                if( (simplex[3].w - simplex[1].w) < NMValueConvergence )
                {
                    terminated = true;
                    break;
                }
            }

            if( terminated || (cycle >= NMSecurityCycleCount && reset) )
            {
                break;
            }
            else
            {
                simplex[0] = vo - glm::dvec4(nodeShift, 0, 0, 0);
                simplex[1] = vo - glm::dvec4(0, nodeShift, 0, 0);
                simplex[2] = vo - glm::dvec4(0, 0, nodeShift, 0);
                simplex[3] = vo;
                reset = true;
                cycle = 0;
            }
        }

        pos = glm::dvec3(simplex[3]);
        if(topo.snapToBoundary->isConstrained())
            pos = (*topo.snapToBoundary)(pos);
    }
}
