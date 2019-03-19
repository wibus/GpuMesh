#ifndef GPUMESH_ABSTRACTTOPOLOGIST
#define GPUMESH_ABSTRACTTOPOLOGIST

#include "DataStructures/OptimizationPlot.h"

class Mesh;
class MeshCrew;
struct Schedule;


class AbstractTopologist
{
protected:
    AbstractTopologist();

public:
    virtual ~AbstractTopologist();


    virtual bool needTopologicalModifications(
            const Mesh& mesh) const = 0;

    virtual void restructureMesh(
            Mesh& mesh,
            const MeshCrew& crew,
            const Schedule& schedule) const = 0;

    virtual void printOptimisationParameters(
            const Mesh& mesh,
            OptimizationPlot& plot) const = 0;

    double minEdgeLength() const;
    void setMinEdgeLength(double length);

    double maxEdgeLength() const;
    void setMaxEdgeLength(double length);


private:
    double _minEdgeLength;
    double _maxEdgeLength;
};



// IMPLEMENTATION //
inline double AbstractTopologist::minEdgeLength() const
{
    return _minEdgeLength;
}

inline double AbstractTopologist::maxEdgeLength() const
{
    return _maxEdgeLength;
}

#endif // GPUMESH_ABSTRACTTOPOLOGIST
