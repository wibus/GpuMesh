#include "BoxBoundary.h"


void installCudaBoxBoundary();


BoxBoundary::BoxBoundary() :
    AbstractBoundary("Box",
        ":/glsl/compute/Boundary/Box.glsl",
        installCudaBoxBoundary),

    _v0(-1, glm::dvec3(-1, -1, -1)),
    _v1(-2, glm::dvec3( 1, -1, -1)),
    _v2(-3, glm::dvec3( 1,  1, -1)),
    _v3(-4, glm::dvec3(-1,  1, -1)),
    _v4(-5, glm::dvec3(-1, -1,  1)),
    _v5(-6, glm::dvec3( 1, -1,  1)),
    _v6(-7, glm::dvec3( 1,  1,  1)),
    _v7(-8, glm::dvec3(-1,  1,  1)),

    _e01(7,  glm::dvec3(-1, -1, -1), glm::dvec3(1, -1, -1)),
    _e23(8,  glm::dvec3(-1,  1, -1), glm::dvec3(1,  1, -1)),
    _e45(9,  glm::dvec3(-1, -1,  1), glm::dvec3(1, -1,  1)),
    _e67(10, glm::dvec3(-1,  1,  1), glm::dvec3(1,  1,  1)),

    _e03(11, glm::dvec3(-1, -1, -1), glm::dvec3(-1,  1, -1)),
    _e12(12, glm::dvec3( 1, -1, -1), glm::dvec3( 1,  1, -1)),
    _e47(13, glm::dvec3(-1, -1,  1), glm::dvec3(-1,  1,  1)),
    _e56(14, glm::dvec3( 1, -1,  1), glm::dvec3( 1,  1,  1)),

    _e04(15, glm::dvec3(-1, -1, -1), glm::dvec3(-1,  -1,  1)),
    _e15(16, glm::dvec3( 1, -1, -1), glm::dvec3( 1,  -1,  1)),
    _e37(17, glm::dvec3(-1,  1, -1), glm::dvec3(-1,   1,  1)),
    _e26(18, glm::dvec3( 1,  1, -1), glm::dvec3( 1,   1,  1)),

    _xNegFace(1, glm::dvec3(-1, 0, 0), glm::dvec3(-1, 0, 0)),
    _xPosFace(2, glm::dvec3( 1, 0, 0), glm::dvec3( 1, 0, 0)),
    _yNegFace(3, glm::dvec3(0, -1, 0), glm::dvec3(0, -1, 0)),
    _yPosFace(4, glm::dvec3(0,  1, 0), glm::dvec3(0,  1, 0)),
    _zNegFace(5, glm::dvec3(0, 0, -1), glm::dvec3(0, 0, -1)),
    _zPosFace(6, glm::dvec3(0, 0,  1), glm::dvec3(0, 0,  1))
{
    // Volume
    volume()->addFace(&_xNegFace);
    volume()->addFace(&_xPosFace);
    volume()->addFace(&_yNegFace);
    volume()->addFace(&_yPosFace);
    volume()->addFace(&_zNegFace);
    volume()->addFace(&_zPosFace);

    // Faces
    _xNegFace.addEdge(&_e03);
    _xNegFace.addEdge(&_e04);
    _xNegFace.addEdge(&_e47);
    _xNegFace.addEdge(&_e37);

    _xPosFace.addEdge(&_e12);
    _xPosFace.addEdge(&_e15);
    _xPosFace.addEdge(&_e56);
    _xPosFace.addEdge(&_e26);

    _yNegFace.addEdge(&_e01);
    _yNegFace.addEdge(&_e04);
    _yNegFace.addEdge(&_e45);
    _yNegFace.addEdge(&_e15);

    _yPosFace.addEdge(&_e37);
    _yPosFace.addEdge(&_e23);
    _yPosFace.addEdge(&_e67);
    _yPosFace.addEdge(&_e26);

    _zNegFace.addEdge(&_e01);
    _zNegFace.addEdge(&_e03);
    _zNegFace.addEdge(&_e23);
    _zNegFace.addEdge(&_e12);

    _zPosFace.addEdge(&_e45);
    _zPosFace.addEdge(&_e47);
    _zPosFace.addEdge(&_e67);
    _zPosFace.addEdge(&_e56);

    // Edges
    _e01.addVertex(&_v0);
    _e01.addVertex(&_v1);
    _e23.addVertex(&_v2);
    _e23.addVertex(&_v3);
    _e45.addVertex(&_v4);
    _e45.addVertex(&_v5);
    _e67.addVertex(&_v6);
    _e67.addVertex(&_v7);

    _e03.addVertex(&_v0);
    _e03.addVertex(&_v3);
    _e12.addVertex(&_v1);
    _e12.addVertex(&_v2);
    _e47.addVertex(&_v4);
    _e47.addVertex(&_v7);
    _e56.addVertex(&_v5);
    _e56.addVertex(&_v6);

    _e04.addVertex(&_v0);
    _e04.addVertex(&_v4);
    _e15.addVertex(&_v1);
    _e15.addVertex(&_v5);
    _e37.addVertex(&_v3);
    _e37.addVertex(&_v7);
    _e26.addVertex(&_v2);
    _e26.addVertex(&_v6);
}

BoxBoundary::~BoxBoundary()
{

}

bool BoxBoundary::unitTest() const
{
    return true;
}
