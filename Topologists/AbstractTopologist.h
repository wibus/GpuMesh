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
            int vertRelocationPassCount,
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

private:
    bool _isEnabled;
    int _frequency;
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

#endif // GPUMESH_ABSTRACTTOPOLOGIST
