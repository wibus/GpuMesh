#ifndef GPUMESH_TET_BOUNDARY
#define GPUMESH_TET_BOUNDARY

#include "AbstractBoundary.h"


class TetBoundary : public AbstractBoundary
{
public:
    TetBoundary();
    virtual ~TetBoundary();


    virtual bool unitTest() const override;


    const AbstractConstraint* v0() const;

    const AbstractConstraint* v1() const;

    const AbstractConstraint* v2() const;

    const AbstractConstraint* v3() const;


    const AbstractConstraint* e01() const;

    const AbstractConstraint* e02() const;

    const AbstractConstraint* e03() const;

    const AbstractConstraint* e12() const;

    const AbstractConstraint* e23() const;

    const AbstractConstraint* e31() const;


    const AbstractConstraint* f021() const;

    const AbstractConstraint* f013() const;

    const AbstractConstraint* f032() const;

    const AbstractConstraint* f123() const;


private:
    //////////////
    // VERTICES //
    //////////////
    VertexConstraint _v0;
    VertexConstraint _v1;
    VertexConstraint _v2;
    VertexConstraint _v3;


    //////////////
    //  EDGES   //
    //////////////
    SegmentConstraint _e01;
    SegmentConstraint _e02;
    SegmentConstraint _e03;
    SegmentConstraint _e12;
    SegmentConstraint _e23;
    SegmentConstraint _e31;

    //////////////
    //  FACES   //
    //////////////
    PlaneConstraint _f021;
    PlaneConstraint _f013;
    PlaneConstraint _f032;
    PlaneConstraint _f123;
};



// IMPLEMENTATION //
inline const AbstractConstraint* TetBoundary::v0() const
{
    return &_v0;
}

inline const AbstractConstraint* TetBoundary::v1() const
{
    return &_v1;
}

inline const AbstractConstraint* TetBoundary::v2() const
{
    return &_v2;
}

inline const AbstractConstraint* TetBoundary::v3() const
{
    return &_v3;
}


inline const AbstractConstraint* TetBoundary::e01() const
{
    return &_e01;
}

inline const AbstractConstraint* TetBoundary::e02() const
{
    return &_e02;
}

inline const AbstractConstraint* TetBoundary::e03() const
{
    return &_e03;
}

inline const AbstractConstraint* TetBoundary::e12() const
{
    return &_e12;
}

inline const AbstractConstraint* TetBoundary::e23() const
{
    return &_e23;
}

inline const AbstractConstraint* TetBoundary::e31() const
{
    return &_e31;
}


inline const AbstractConstraint* TetBoundary::f021() const
{
    return &_f021;
}

inline const AbstractConstraint* TetBoundary::f013() const
{
    return &_f013;
}

inline const AbstractConstraint* TetBoundary::f032() const
{
    return &_f032;
}

inline const AbstractConstraint* TetBoundary::f123() const
{
    return &_f123;
}

#endif // GPUMESH_TET_BOUNDARY
