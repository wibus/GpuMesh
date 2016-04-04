#include "NelderMeadSmoother.h"

#include <limits>

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"

using namespace std;
using namespace cellar;


// CUDA Drivers
void installCudaNelderMeadSmoother();


NelderMeadSmoother::NelderMeadSmoother() :
    AbstractVertexWiseSmoother(
        {":/glsl/compute/Smoothing/VertexWise/NelderMead.glsl"},
        installCudaNelderMeadSmoother),
    _securityCycleCount(12),
    _localSizeToNodeShift(1.0 / 24.0),
    _alpha(1.0),
    _beta(0.5),
    _gamma(2.0),
    _delta(0.5)
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
    program.setFloat("LocalSizeToNodeShift", _localSizeToNodeShift);
    program.setInt("SecurityCycleCount", _securityCycleCount);
    program.setInt("GainThreshold", _gainThreshold);
    program.setFloat("Alpha", _alpha);
    program.setFloat("Beta", _beta);
    program.setFloat("Gamma", _gamma);
    program.setFloat("Delta", _delta);
}

void NelderMeadSmoother::printSmoothingParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{
    AbstractVertexWiseSmoother::printSmoothingParameters(mesh, plot);
    plot.addSmoothingProperty("Method Name", "Local Optimization");
    plot.addSmoothingProperty("Local Size to Node Shift", to_string(_localSizeToNodeShift));
    plot.addSmoothingProperty("Security Cycle Count", to_string(_securityCycleCount));
    plot.addSmoothingProperty("Gain Threshold", to_string(_gainThreshold));
    plot.addSmoothingProperty("Reflexion", to_string(_alpha));
    plot.addSmoothingProperty("Contraction", to_string(_beta));
    plot.addSmoothingProperty("Expansion", to_string(_gamma));
    plot.addSmoothingProperty("Shrinkage", to_string(_delta));
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

        if(!isSmoothable(mesh, vId))
            continue;


        // Compute local element size
        double localSize =
            crew.measurer().computeLocalElementSize(
                mesh, crew.sampler(), vId);


        // Initialize node shift distance
        double nodeShift = localSize * _localSizeToNodeShift;


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
                if(topo.isBoundary)
                    pos = (*topo.snapToBoundary)(glm::dvec3(simplex[p]));
                else
                    pos = glm::dvec3(simplex[p]);


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


            for(; cycle < _securityCycleCount; ++cycle)
            {
                // Centroid
                glm::dvec3 c = 1/3.0 * (
                    glm::dvec3(simplex[1]) +
                    glm::dvec3(simplex[2]) +
                    glm::dvec3(simplex[3]));

                double f = 0.0;

                // Reflect
                pos = c + _alpha*(c - glm::dvec3(simplex[0]));
                if(topo.isBoundary) pos = (*topo.snapToBoundary)(pos);
                double fr = f = crew.evaluator().patchQuality(
                    mesh, crew.sampler(), crew.measurer(), vId);

                glm::dvec3 xr = pos;

                // Expand
                if(simplex[3].w < fr)
                {
                    pos = c + _gamma*(pos - c);
                    if(topo.isBoundary) pos = (*topo.snapToBoundary)(pos);
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
                        pos = c + _beta*(xr - c);
                        if(topo.isBoundary) pos = (*topo.snapToBoundary)(pos);
                        f = crew.evaluator().patchQuality(
                            mesh, crew.sampler(), crew.measurer(), vId);
                    }
                    // Inside
                    else
                    {
                        pos = c + _beta*(glm::dvec3(simplex[0]) - c);
                        if(topo.isBoundary) pos = (*topo.snapToBoundary)(pos);
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


                if( (simplex[3].w - simplex[1].w) < _gainThreshold )
                {
                    terminated = true;
                    break;
                }
            }

            if( terminated || (cycle >= _securityCycleCount && reset) )
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

        if(topo.isBoundary)
            pos = (*topo.snapToBoundary)(glm::dvec3(simplex[3]));
        else
            pos = glm::dvec3(simplex[3]);
    }
}
