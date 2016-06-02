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


    virtual bool needTopologicalModifications(
            const Mesh& mesh) const = 0;

    virtual void restructureMesh(
            Mesh& mesh,
            const MeshCrew& crew) const = 0;

    virtual void printOptimisationParameters(
            const Mesh& mesh,
            OptimizationPlot& plot) const = 0;


    bool isEnabled() const;
    void setEnabled(bool isEnabled);

    int frequency() const;
    void setFrequency(int frequency);

    int topoPassCount() const;
    void setTopoPassCount(int count);

    double minEdgeLength() const;
    void setMinEdgeLength(double length);

    double maxEdgeLength() const;
    void setMaxEdgeLength(double length);


private:
    bool _isEnabled;
    int _frequency;
    int _topoPassCount;
    double _minEdgeLength;
    double _maxEdgeLength;
};



// IMPLEMENTATION //
inline bool AbstractTopologist::isEnabled() const
{
    return _isEnabled;
}

inline int AbstractTopologist::frequency() const
{
    return _frequency;
}

inline int AbstractTopologist::topoPassCount() const
{
    return _topoPassCount;
}

inline double AbstractTopologist::minEdgeLength() const
{
    return _minEdgeLength;
}

inline double AbstractTopologist::maxEdgeLength() const
{
    return _maxEdgeLength;
}

#endif // GPUMESH_ABSTRACTTOPOLOGIST
