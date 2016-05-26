#ifndef GPUMESH_BOX_BOUNDARY
#define GPUMESH_BOX_BOUNDARY

#include "Boundary.h"


class BoxBoundary : public MeshBoundary
{
public:
    BoxBoundary();
    virtual ~BoxBoundary();


    const TopologyConstraint* XnYnZnVertex() const;

    const TopologyConstraint* XpYnZnVertex() const;

    const TopologyConstraint* XpYpZnVertex() const;

    const TopologyConstraint* XnYpZnVertex() const;

    const TopologyConstraint* XnYnZpVertex() const;

    const TopologyConstraint* XpYnZpVertex() const;

    const TopologyConstraint* XpYpZpVertex() const;

    const TopologyConstraint* XnYpZpVertex() const;


    const TopologyConstraint* xNyNzEdge() const;

    const TopologyConstraint* xPyNzEdge() const;

    const TopologyConstraint* xNyPzEdge() const;

    const TopologyConstraint* xPyPzEdge() const;


    const TopologyConstraint* yNxNzEdge() const;

    const TopologyConstraint* yPxNzEdge() const;

    const TopologyConstraint* yNxPzEdge() const;

    const TopologyConstraint* yPxPzEdge() const;


    const TopologyConstraint* zNxNxEdge() const;

    const TopologyConstraint* zPxNxEdge() const;

    const TopologyConstraint* zNxPxEdge() const;

    const TopologyConstraint* zPxPxEdge() const;


    const TopologyConstraint* xNegSurface() const;

    const TopologyConstraint* xPosSurface() const;

    const TopologyConstraint* yNegSurface() const;

    const TopologyConstraint* yPosSurface() const;

    const TopologyConstraint* zNegSurface() const;

    const TopologyConstraint* zPosSurface() const;


    const TopologyConstraint* volume() const;

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
inline const TopologyConstraint* BoxBoundary::XnYnZnVertex() const
{
    return &_XnYnZnVertex;
}

inline const TopologyConstraint* BoxBoundary::XpYnZnVertex() const
{
    return &_XpYnZnVertex;
}

inline const TopologyConstraint* BoxBoundary::XpYpZnVertex() const
{
    return &_XpYpZnVertex;
}

inline const TopologyConstraint* BoxBoundary::XnYpZnVertex() const
{
    return &_XnYpZnVertex;
}

inline const TopologyConstraint* BoxBoundary::XnYnZpVertex() const
{
    return &_XnYnZpVertex;
}

inline const TopologyConstraint* BoxBoundary::XpYnZpVertex() const
{
    return &_XpYnZpVertex;
}

inline const TopologyConstraint* BoxBoundary::XpYpZpVertex() const
{
    return &_XpYpZpVertex;
}

inline const TopologyConstraint* BoxBoundary::XnYpZpVertex() const
{
    return &_XnYpZpVertex;
}


inline const TopologyConstraint* BoxBoundary::xNyNzEdge() const
{
    return &_xNyNzEdge;
}

inline const TopologyConstraint* BoxBoundary::xPyNzEdge() const
{
    return &_xPyNzEdge;
}

inline const TopologyConstraint* BoxBoundary::xNyPzEdge() const
{
    return &_xNyPzEdge;
}

inline const TopologyConstraint* BoxBoundary::xPyPzEdge() const
{
    return &_xPyPzEdge;
}


inline const TopologyConstraint* BoxBoundary::yNxNzEdge() const
{
    return &_yNxNzEdge;
}

inline const TopologyConstraint* BoxBoundary::yPxNzEdge() const
{
    return &_yPxNzEdge;
}

inline const TopologyConstraint* BoxBoundary::yNxPzEdge() const
{
    return &_yNxPzEdge;
}

inline const TopologyConstraint* BoxBoundary::yPxPzEdge() const
{
    return &_yPxPzEdge;
}


inline const TopologyConstraint* BoxBoundary::zNxNxEdge() const
{
    return &_zNxNyEdge;
}

inline const TopologyConstraint* BoxBoundary::zPxNxEdge() const
{
    return &_zPxNyEdge;
}

inline const TopologyConstraint* BoxBoundary::zNxPxEdge() const
{
    return &_zNxPyEdge;
}

inline const TopologyConstraint* BoxBoundary::zPxPxEdge() const
{
    return &_zPxPyEdge;
}


inline const TopologyConstraint* BoxBoundary::xNegSurface() const
{
    return &_xNegSurface;
}

inline const TopologyConstraint* BoxBoundary::xPosSurface() const
{
    return &_xPosSurface;
}

inline const TopologyConstraint* BoxBoundary::yNegSurface() const
{
    return &_yNegSurface;
}

inline const TopologyConstraint* BoxBoundary::yPosSurface() const
{
    return &_yPosSurface;
}

inline const TopologyConstraint* BoxBoundary::zNegSurface() const
{
    return &_zNegSurface;
}

inline const TopologyConstraint* BoxBoundary::zPosSurface() const
{
    return &_zPosSurface;
}

#endif // GPUMESH_BOX_BOUNDARY
