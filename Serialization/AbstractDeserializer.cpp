#include "AbstractDeserializer.h"

#include "Boundaries/BoundaryFree.h"
#include "Boundaries/BoxBoundary.h"
#include "Boundaries/PipeBoundary.h"
#include "Boundaries/ShellBoundary.h"
#include "Boundaries/SphereBoundary.h"
#include "Boundaries/TetBoundary.h"

using namespace std;


AbstractDeserializer::AbstractDeserializer() :
    _boundaries("Boundaries")
{
    _boundaries.setDefault(BoundaryFree().name());
    _boundaries.setContent({
       {BoundaryFree().name(),      shared_ptr<AbstractBoundary>(new BoundaryFree())},
       {BoxBoundary().name(),       shared_ptr<AbstractBoundary>(new BoxBoundary())},
       {PipeBoundary().name(),      shared_ptr<AbstractBoundary>(new PipeBoundary())},
       {ShellBoundary().name(),     shared_ptr<AbstractBoundary>(new ShellBoundary())},
       {SphereBoundary().name(),    shared_ptr<AbstractBoundary>(new SphereBoundary())},
       {TetBoundary().name(),       shared_ptr<AbstractBoundary>(new TetBoundary())}
    });
}

AbstractDeserializer::~AbstractDeserializer()
{

}

shared_ptr<AbstractBoundary> AbstractDeserializer::boundary(const string& name) const
{
    std::shared_ptr<AbstractBoundary> bound;
    _boundaries.select(name, bound);
    return bound;
}
