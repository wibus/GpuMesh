#ifndef GPUMESH_ABSTRACTTOPOLOGIST
#define GPUMESH_ABSTRACTTOPOLOGIST

#include "DataStructures/OptimizationPlot.h"

class Mesh;
class MeshCrew;


class AbstractTopologist
{
protected:
    AbstractTopologist();

public:
    virtual ~AbstractTopologist();

    virtual void restructureMesh(
            Mesh& mesh,
            const MeshCrew& crew) = 0;

    virtual void printOptimisationParameters(
            const Mesh& mesh,
            OptimizationPlot& plot) const = 0;
};

#endif // GPUMESH_ABSTRACTTOPOLOGIST
