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
    _XnYnZnVertex.addEdge(&_xNyNzEdge);
    _XnYnZnVertex.addEdge(&_yNxNzEdge);
    _XnYnZnVertex.addEdge(&_zNxNyEdge);
    addVertex(&_XnYnZnVertex);
}

BoxBoundary::~BoxBoundary()
{

}
