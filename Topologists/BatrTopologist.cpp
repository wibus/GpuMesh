#include "BatrTopologist.h"

#include <CellarWorkbench/Misc/Log.h>

using namespace cellar;


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
}

void BatrTopologist::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{

}
