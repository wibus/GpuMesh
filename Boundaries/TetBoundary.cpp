#include "TetBoundary.h"

const glm::dvec3 TET_FIRST_MOMENT = glm::dvec3(0.5, 0.288675, 0.204124);

TetBoundary::TetBoundary() :
    AbstractBoundary("Tet"),

    _v0(-1, 2.0 * (glm::dvec3(0, 0, 0) - TET_FIRST_MOMENT)),
    _v1(-2, 2.0 * (glm::dvec3(1, 0, 0) - TET_FIRST_MOMENT)),
    _v2(-3, 2.0 * (glm::dvec3(0.5, sqrt(3.0)/2, 0) - TET_FIRST_MOMENT)),
    _v3(-4, 2.0 * (glm::dvec3(0.5, sqrt(3.0)/6, sqrt(2.0/3)) - TET_FIRST_MOMENT)),

    _e01(7,  _v0.position(), _v1.position()),
    _e02(8,  _v0.position(), _v2.position()),
    _e03(9,  _v0.position(), _v3.position()),
    _e12(10, _v1.position(), _v2.position()),
    _e23(11, _v2.position(), _v3.position()),
    _e31(12, _v3.position(), _v1.position()),

    _f021(1, (_v0.position() + _v2.position() + _v1.position()) / 3.0, glm::cross(_e02.direction(), -_e12.direction())),
    _f013(2, (_v0.position() + _v1.position() + _v3.position()) / 3.0, glm::cross(_e01.direction(), -_e31.direction())),
    _f032(3, (_v0.position() + _v3.position() + _v2.position()) / 3.0, glm::cross(_e03.direction(), -_e23.direction())),
    _f123(4, (_v1.position() + _v2.position() + _v3.position()) / 3.0, glm::cross(_e12.direction(),  _e23.direction()))
{
    // Volume
    volume()->addFace(&_f021);
    volume()->addFace(&_f013);
    volume()->addFace(&_f032);
    volume()->addFace(&_f123);

    // Faces
    _f021.addEdge(&_e02);
    _f021.addEdge(&_e12);
    _f021.addEdge(&_e01);

    _f013.addEdge(&_e01);
    _f013.addEdge(&_e31);
    _f013.addEdge(&_e03);

    _f032.addEdge(&_e03);
    _f032.addEdge(&_e23);
    _f032.addEdge(&_e02);

    _f123.addEdge(&_e12);
    _f123.addEdge(&_e23);
    _f123.addEdge(&_e31);


    // Edges
    _e01.addVertex(&_v0);
    _e01.addVertex(&_v1);

    _e02.addVertex(&_v0);
    _e02.addVertex(&_v2);

    _e03.addVertex(&_v0);
    _e03.addVertex(&_v3);

    _e12.addVertex(&_v1);
    _e12.addVertex(&_v2);

    _e23.addVertex(&_v2);
    _e23.addVertex(&_v3);

    _e31.addVertex(&_v3);
    _e31.addVertex(&_v1);
}

TetBoundary::~TetBoundary()
{

}

bool TetBoundary::unitTest() const
{
    return true;
}
