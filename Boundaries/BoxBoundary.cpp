#include "BoxBoundary.h"


BoxBoundary::BoxBoundary() :
    _XnYnZnVertex(-1, glm::dvec3(-1, -1, -1)),
    _XpYnZnVertex(-2, glm::dvec3( 1, -1, -1)),
    _XpYpZnVertex(-3, glm::dvec3( 1,  1, -1)),
    _XnYpZnVertex(-4, glm::dvec3(-1,  1, -1)),
    _XnYnZpVertex(-5, glm::dvec3(-1, -1,  1)),
    _XpYnZpVertex(-6, glm::dvec3( 1, -1,  1)),
    _XpYpZpVertex(-7, glm::dvec3( 1,  1,  1)),
    _XnYpZpVertex(-8, glm::dvec3(-1,  1,  1)),

    _xNyNzEdge(7,  glm::dvec3(-1, -1, -1), glm::dvec3(1, -1, -1)),
    _xPyNzEdge(8,  glm::dvec3(-1,  1, -1), glm::dvec3(1,  1, -1)),
    _xNyPzEdge(9,  glm::dvec3(-1, -1,  1), glm::dvec3(1, -1,  1)),
    _xPyPzEdge(10, glm::dvec3(-1,  1,  1), glm::dvec3(1,  1,  1)),

    _yNxNzEdge(11, glm::dvec3(-1, -1, -1), glm::dvec3(-1,  1, -1)),
    _yPxNzEdge(12, glm::dvec3( 1, -1, -1), glm::dvec3( 1,  1, -1)),
    _yNxPzEdge(13, glm::dvec3(-1, -1,  1), glm::dvec3(-1,  1,  1)),
    _yPxPzEdge(14, glm::dvec3( 1, -1,  1), glm::dvec3( 1,  1,  1)),

    _zNxNyEdge(15, glm::dvec3(-1, -1, -1), glm::dvec3(-1,  -1,  1)),
    _zPxNyEdge(16, glm::dvec3( 1, -1, -1), glm::dvec3( 1,  -1,  1)),
    _zNxPyEdge(17, glm::dvec3(-1,  1, -1), glm::dvec3(-1,   1,  1)),
    _zPxPyEdge(18, glm::dvec3( 1,  1, -1), glm::dvec3( 1,   1,  1)),

    _xNegSurface(1, glm::dvec3(-1, 0, 0), glm::dvec3(-1, 0, 0)),
    _xPosSurface(2, glm::dvec3( 1, 0, 0), glm::dvec3( 1, 0, 0)),
    _yNegSurface(3, glm::dvec3(0, -1, 0), glm::dvec3(0, -1, 0)),
    _yPosSurface(4, glm::dvec3(0,  1, 0), glm::dvec3(0,  1, 0)),
    _zNegSurface(5, glm::dvec3(0, 0, -1), glm::dvec3(0, 0, -1)),
    _zPosSurface(6, glm::dvec3(0, 0,  1), glm::dvec3(0, 0,  1))
{
    // Volume
    volume()->addSurface(&_xNegSurface);
    volume()->addSurface(&_xPosSurface);
    volume()->addSurface(&_yNegSurface);
    volume()->addSurface(&_yPosSurface);
    volume()->addSurface(&_zNegSurface);
    volume()->addSurface(&_zPosSurface);

    // Surfaces
    _xNegSurface.addEdge(&_yNxNzEdge);
    _xNegSurface.addEdge(&_zNxNyEdge);
    _xNegSurface.addEdge(&_yNxPzEdge);
    _xNegSurface.addEdge(&_zNxPyEdge);

    _xPosSurface.addEdge(&_yPxNzEdge);
    _xPosSurface.addEdge(&_zPxNyEdge);
    _xPosSurface.addEdge(&_yPxPzEdge);
    _xPosSurface.addEdge(&_zPxPyEdge);

    _yNegSurface.addEdge(&_xNyNzEdge);
    _yNegSurface.addEdge(&_zNxNyEdge);
    _yNegSurface.addEdge(&_xNyPzEdge);
    _yNegSurface.addEdge(&_zPxNyEdge);

    _yPosSurface.addEdge(&_zNxPyEdge);
    _yPosSurface.addEdge(&_xPyNzEdge);
    _yPosSurface.addEdge(&_xPyPzEdge);
    _yPosSurface.addEdge(&_zPxPyEdge);

    _zNegSurface.addEdge(&_xNyNzEdge);
    _zNegSurface.addEdge(&_yNxNzEdge);
    _zNegSurface.addEdge(&_xPyNzEdge);
    _zNegSurface.addEdge(&_yPxNzEdge);

    _zPosSurface.addEdge(&_xNyPzEdge);
    _zPosSurface.addEdge(&_yNxPzEdge);
    _zPosSurface.addEdge(&_xPyPzEdge);
    _zPosSurface.addEdge(&_yPxPzEdge);

    // Edges
    _xNyNzEdge.addVertex(&_XnYnZnVertex);
    _xNyNzEdge.addVertex(&_XpYnZnVertex);
    _xPyNzEdge.addVertex(&_XnYpZnVertex);
    _xPyNzEdge.addVertex(&_XpYpZnVertex);
    _xNyPzEdge.addVertex(&_XnYnZpVertex);
    _xNyPzEdge.addVertex(&_XpYnZpVertex);
    _xPyPzEdge.addVertex(&_XnYpZpVertex);
    _xPyPzEdge.addVertex(&_XpYpZpVertex);

    _yNxNzEdge.addVertex(&_XnYnZnVertex);
    _yNxNzEdge.addVertex(&_XnYpZnVertex);
    _yPxNzEdge.addVertex(&_XpYnZnVertex);
    _yPxNzEdge.addVertex(&_XpYpZnVertex);
    _yNxPzEdge.addVertex(&_XnYnZpVertex);
    _yNxPzEdge.addVertex(&_XnYpZpVertex);
    _yPxPzEdge.addVertex(&_XpYnZnVertex);
    _yPxPzEdge.addVertex(&_XpYpZpVertex);

    _zNxNyEdge.addVertex(&_XnYnZnVertex);
    _zNxNyEdge.addVertex(&_XnYnZpVertex);
    _zPxNyEdge.addVertex(&_XpYnZnVertex);
    _zPxNyEdge.addVertex(&_XpYnZpVertex);
    _zNxPyEdge.addVertex(&_XnYpZnVertex);
    _zNxPyEdge.addVertex(&_XnYpZpVertex);
    _zPxPyEdge.addVertex(&_XpYpZnVertex);
    _zPxPyEdge.addVertex(&_XpYpZpVertex);
}

BoxBoundary::~BoxBoundary()
{

}
