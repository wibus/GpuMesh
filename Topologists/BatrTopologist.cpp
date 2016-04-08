#include "BatrTopologist.h"

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/Mesh.h"

using namespace cellar;

extern bool verboseCuda;


BatrTopologist::BatrTopologist()
{

}

BatrTopologist::~BatrTopologist()
{

}

void BatrTopologist::restructureMesh(
        Mesh& mesh,
        const MeshCrew& crew) const
{
    getLog().postMessage(new Message('I', false,
        "Performing new BATR topology modifications",
        "BatrTopologist"));

    edgeSplitting(mesh, crew);
    faceSwapping(mesh, crew);
    edgeSwapping(mesh, crew);

    verboseCuda = false;
    mesh.compileTopology(false);
    verboseCuda = true;
}

void BatrTopologist::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{

}


void BatrTopologist::edgeSplitting(
        Mesh& mesh,
        const MeshCrew& crew) const
{

}

void BatrTopologist::faceSwapping(
        Mesh& mesh,
        const MeshCrew& crew) const
{

}

void BatrTopologist::edgeSwapping(
        Mesh& mesh,
        const MeshCrew& crew) const
{

}
