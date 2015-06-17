#ifndef GPUMESH_ABSTRACTMESHER
#define GPUMESH_ABSTRACTMESHER

#include <string>

#include "DataStructures/Mesh.h"


class AbstractMesher
{
public:
    AbstractMesher(Mesh& mesh, unsigned int vertCount);
    virtual ~AbstractMesher();

    bool processFinished() const;

    virtual void resetPipeline();
    virtual void processPipeline();
    virtual void scheduleSmoothing();


protected:
    virtual void genBoundaryMeshes();
    virtual void triangulateDomain();
    virtual void computeAdjacency();
    virtual void smoothMesh();

    virtual void printStep(int step, const std::string& stepName);

protected:
    Mesh& _mesh;
    unsigned int _vertCount;

    bool _processFinished;
    int _stepId;
};

#endif // GPUMESH_ABSTRACTMESHER
