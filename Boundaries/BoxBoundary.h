#ifndef GPUMESH_BOX_BOUNDARY
#define GPUMESH_BOX_BOUNDARY

#include "AbstractBoundary.h"


class BoxBoundary : public AbstractBoundary
{
public:
    BoxBoundary();
    virtual ~BoxBoundary();


    virtual bool unitTest() const override;


    const AbstractConstraint* v0() const;

    const AbstractConstraint* v1() const;

    const AbstractConstraint* v2() const;

    const AbstractConstraint* v3() const;

    const AbstractConstraint* v4() const;

    const AbstractConstraint* v5() const;

    const AbstractConstraint* v6() const;

    const AbstractConstraint* v7() const;


    const AbstractConstraint* e01() const;

    const AbstractConstraint* e23() const;

    const AbstractConstraint* e45() const;

    const AbstractConstraint* e67() const;


    const AbstractConstraint* e03() const;

    const AbstractConstraint* e12() const;

    const AbstractConstraint* e47() const;

    const AbstractConstraint* e56() const;


    const AbstractConstraint* e04() const;

    const AbstractConstraint* e15() const;

    const AbstractConstraint* e37() const;

    const AbstractConstraint* e26() const;


    const AbstractConstraint* xNegFace() const;

    const AbstractConstraint* xPosFace() const;

    const AbstractConstraint* yNegFace() const;

    const AbstractConstraint* yPosFace() const;

    const AbstractConstraint* zNegFace() const;

    const AbstractConstraint* zPosFace() const;


private:
    //////////////
    // VERTICES //
    //////////////
    VertexConstraint _v0;
    VertexConstraint _v1;
    VertexConstraint _v2;
    VertexConstraint _v3;
    VertexConstraint _v4;
    VertexConstraint _v5;
    VertexConstraint _v6;
    VertexConstraint _v7;


    //////////////
    //  EDGES   //
    //////////////
    SegmentConstraint _e01;
    SegmentConstraint _e23;
    SegmentConstraint _e45;
    SegmentConstraint _e67;

    SegmentConstraint _e03;
    SegmentConstraint _e12;
    SegmentConstraint _e47;
    SegmentConstraint _e56;

    SegmentConstraint _e04;
    SegmentConstraint _e15;
    SegmentConstraint _e37;
    SegmentConstraint _e26;


    //////////////
    //  FACES   //
    //////////////
    PlaneConstraint _xNegFace;
    PlaneConstraint _xPosFace;
    PlaneConstraint _yNegFace;
    PlaneConstraint _yPosFace;
    PlaneConstraint _zNegFace;
    PlaneConstraint _zPosFace;
};



// IMPLEMENTATION //
inline const AbstractConstraint* BoxBoundary::v0() const
{
    return &_v0;
}

inline const AbstractConstraint* BoxBoundary::v1() const
{
    return &_v1;
}

inline const AbstractConstraint* BoxBoundary::v2() const
{
    return &_v2;
}

inline const AbstractConstraint* BoxBoundary::v3() const
{
    return &_v3;
}

inline const AbstractConstraint* BoxBoundary::v4() const
{
    return &_v4;
}

inline const AbstractConstraint* BoxBoundary::v5() const
{
    return &_v5;
}

inline const AbstractConstraint* BoxBoundary::v6() const
{
    return &_v6;
}

inline const AbstractConstraint* BoxBoundary::v7() const
{
    return &_v7;
}


inline const AbstractConstraint* BoxBoundary::e01() const
{
    return &_e01;
}

inline const AbstractConstraint* BoxBoundary::e23() const
{
    return &_e23;
}

inline const AbstractConstraint* BoxBoundary::e45() const
{
    return &_e45;
}

inline const AbstractConstraint* BoxBoundary::e67() const
{
    return &_e67;
}


inline const AbstractConstraint* BoxBoundary::e03() const
{
    return &_e03;
}

inline const AbstractConstraint* BoxBoundary::e12() const
{
    return &_e12;
}

inline const AbstractConstraint* BoxBoundary::e47() const
{
    return &_e47;
}

inline const AbstractConstraint* BoxBoundary::e56() const
{
    return &_e56;
}


inline const AbstractConstraint* BoxBoundary::e04() const
{
    return &_e04;
}

inline const AbstractConstraint* BoxBoundary::e15() const
{
    return &_e15;
}

inline const AbstractConstraint* BoxBoundary::e37() const
{
    return &_e37;
}

inline const AbstractConstraint* BoxBoundary::e26() const
{
    return &_e26;
}


inline const AbstractConstraint* BoxBoundary::xNegFace() const
{
    return &_xNegFace;
}

inline const AbstractConstraint* BoxBoundary::xPosFace() const
{
    return &_xPosFace;
}

inline const AbstractConstraint* BoxBoundary::yNegFace() const
{
    return &_yNegFace;
}

inline const AbstractConstraint* BoxBoundary::yPosFace() const
{
    return &_yPosFace;
}

inline const AbstractConstraint* BoxBoundary::zNegFace() const
{
    return &_zNegFace;
}

inline const AbstractConstraint* BoxBoundary::zPosFace() const
{
    return &_zPosFace;
}

#endif // GPUMESH_BOX_BOUNDARY
