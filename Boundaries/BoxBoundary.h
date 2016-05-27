#ifndef GPUMESH_BOX_BOUNDARY
#define GPUMESH_BOX_BOUNDARY

#include "AbstractBoundary.h"


class BoxBoundary : public AbstractBoundary
{
public:
    BoxBoundary();
    virtual ~BoxBoundary();


    const AbstractConstraint* XnYnZnVertex() const;

    const AbstractConstraint* XpYnZnVertex() const;

    const AbstractConstraint* XpYpZnVertex() const;

    const AbstractConstraint* XnYpZnVertex() const;

    const AbstractConstraint* XnYnZpVertex() const;

    const AbstractConstraint* XpYnZpVertex() const;

    const AbstractConstraint* XpYpZpVertex() const;

    const AbstractConstraint* XnYpZpVertex() const;


    const AbstractConstraint* xNyNzEdge() const;

    const AbstractConstraint* xPyNzEdge() const;

    const AbstractConstraint* xNyPzEdge() const;

    const AbstractConstraint* xPyPzEdge() const;


    const AbstractConstraint* yNxNzEdge() const;

    const AbstractConstraint* yPxNzEdge() const;

    const AbstractConstraint* yNxPzEdge() const;

    const AbstractConstraint* yPxPzEdge() const;


    const AbstractConstraint* zNxNxEdge() const;

    const AbstractConstraint* zPxNxEdge() const;

    const AbstractConstraint* zNxPxEdge() const;

    const AbstractConstraint* zPxPxEdge() const;


    const AbstractConstraint* xNegSurface() const;

    const AbstractConstraint* xPosSurface() const;

    const AbstractConstraint* yNegSurface() const;

    const AbstractConstraint* yPosSurface() const;

    const AbstractConstraint* zNegSurface() const;

    const AbstractConstraint* zPosSurface() const;

private:
    //////////////
    // VERTICES //
    //////////////
    VertexConstraint _XnYnZnVertex;
    VertexConstraint _XpYnZnVertex;
    VertexConstraint _XpYpZnVertex;
    VertexConstraint _XnYpZnVertex;
    VertexConstraint _XnYnZpVertex;
    VertexConstraint _XpYnZpVertex;
    VertexConstraint _XpYpZpVertex;
    VertexConstraint _XnYpZpVertex;


    //////////////
    //  EDGES   //
    //////////////
    SegmentConstraint _xNyNzEdge;
    SegmentConstraint _xPyNzEdge;
    SegmentConstraint _xNyPzEdge;
    SegmentConstraint _xPyPzEdge;

    SegmentConstraint _yNxNzEdge;
    SegmentConstraint _yPxNzEdge;
    SegmentConstraint _yNxPzEdge;
    SegmentConstraint _yPxPzEdge;

    SegmentConstraint _zNxNyEdge;
    SegmentConstraint _zPxNyEdge;
    SegmentConstraint _zNxPyEdge;
    SegmentConstraint _zPxPyEdge;


    //////////////
    // SURFACES //
    //////////////
    PlaneConstraint _xNegSurface;
    PlaneConstraint _xPosSurface;
    PlaneConstraint _yNegSurface;
    PlaneConstraint _yPosSurface;
    PlaneConstraint _zNegSurface;
    PlaneConstraint _zPosSurface;
};



// IMPLEMENTATION //
inline const AbstractConstraint* BoxBoundary::XnYnZnVertex() const
{
    return &_XnYnZnVertex;
}

inline const AbstractConstraint* BoxBoundary::XpYnZnVertex() const
{
    return &_XpYnZnVertex;
}

inline const AbstractConstraint* BoxBoundary::XpYpZnVertex() const
{
    return &_XpYpZnVertex;
}

inline const AbstractConstraint* BoxBoundary::XnYpZnVertex() const
{
    return &_XnYpZnVertex;
}

inline const AbstractConstraint* BoxBoundary::XnYnZpVertex() const
{
    return &_XnYnZpVertex;
}

inline const AbstractConstraint* BoxBoundary::XpYnZpVertex() const
{
    return &_XpYnZpVertex;
}

inline const AbstractConstraint* BoxBoundary::XpYpZpVertex() const
{
    return &_XpYpZpVertex;
}

inline const AbstractConstraint* BoxBoundary::XnYpZpVertex() const
{
    return &_XnYpZpVertex;
}


inline const AbstractConstraint* BoxBoundary::xNyNzEdge() const
{
    return &_xNyNzEdge;
}

inline const AbstractConstraint* BoxBoundary::xPyNzEdge() const
{
    return &_xPyNzEdge;
}

inline const AbstractConstraint* BoxBoundary::xNyPzEdge() const
{
    return &_xNyPzEdge;
}

inline const AbstractConstraint* BoxBoundary::xPyPzEdge() const
{
    return &_xPyPzEdge;
}


inline const AbstractConstraint* BoxBoundary::yNxNzEdge() const
{
    return &_yNxNzEdge;
}

inline const AbstractConstraint* BoxBoundary::yPxNzEdge() const
{
    return &_yPxNzEdge;
}

inline const AbstractConstraint* BoxBoundary::yNxPzEdge() const
{
    return &_yNxPzEdge;
}

inline const AbstractConstraint* BoxBoundary::yPxPzEdge() const
{
    return &_yPxPzEdge;
}


inline const AbstractConstraint* BoxBoundary::zNxNxEdge() const
{
    return &_zNxNyEdge;
}

inline const AbstractConstraint* BoxBoundary::zPxNxEdge() const
{
    return &_zPxNyEdge;
}

inline const AbstractConstraint* BoxBoundary::zNxPxEdge() const
{
    return &_zNxPyEdge;
}

inline const AbstractConstraint* BoxBoundary::zPxPxEdge() const
{
    return &_zPxPyEdge;
}


inline const AbstractConstraint* BoxBoundary::xNegSurface() const
{
    return &_xNegSurface;
}

inline const AbstractConstraint* BoxBoundary::xPosSurface() const
{
    return &_xPosSurface;
}

inline const AbstractConstraint* BoxBoundary::yNegSurface() const
{
    return &_yNegSurface;
}

inline const AbstractConstraint* BoxBoundary::yPosSurface() const
{
    return &_yPosSurface;
}

inline const AbstractConstraint* BoxBoundary::zNegSurface() const
{
    return &_zNegSurface;
}

inline const AbstractConstraint* BoxBoundary::zPosSurface() const
{
    return &_zPosSurface;
}

#endif // GPUMESH_BOX_BOUNDARY
