#ifndef GPUMESH_ABSTRACTMESHER
#define GPUMESH_ABSTRACTMESHER

#include <string>

#include "DataStructures/Mesh.h"


class AbstractMesher
{
public:
    AbstractMesher(Mesh& mesh, unsigned int vertCount);
    virtual ~AbstractMesher() = 0;

    bool processFinished() const;

    virtual void resetPipeline();
    virtual void processPipeline();
    virtual void scheduleSmoothing();


protected:
    virtual void printStep(int step, const std::string& stepName);
    virtual void triangulateDomain();
    virtual void smoothMesh();


protected:
    Mesh& _mesh;
    unsigned int _vertCount;

    int _stepId;
    bool _processFinished;
};

#endif // GPUMESH_ABSTRACTMESHER
