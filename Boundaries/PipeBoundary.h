#ifndef GPUMESH_PIPE_BOUNDARY
#define GPUMESH_PIPE_BOUNDARY

#include "AbstractBoundary.h"


class PipeBoundary : public AbstractBoundary
{
public:
    PipeBoundary();
    virtual ~PipeBoundary();


    virtual bool unitTest() const override;


    const FaceConstraint* cylinderFace() const;

    const FaceConstraint* yNegDiskFace() const;

    const FaceConstraint* yPosDiskFace() const;

    const EdgeConstraint* yNegCircleEdge() const;

    const EdgeConstraint* yPosCircleEdge() const;


    static const double PIPE_RADIUS;
    static const glm::dvec3 EXT_FACE_NORMAL;
    static const glm::dvec3 EXT_YNEG_CENTER;
    static const glm::dvec3 EXT_YPOS_CENTER;


private:
    class CylinderFace : public FaceConstraint
    {
    public:
        CylinderFace();
        virtual glm::dvec3 operator()(
            const glm::dvec3 &pos) const override;
    } _cylinderFace;


    class YNegDiskFace : public FaceConstraint
    {
    public:
        YNegDiskFace();
        virtual glm::dvec3 operator()(
            const glm::dvec3 &pos) const override;
    } _yNegDiskFace;

    class YPosDiskFace : public FaceConstraint
    {
    public:
        YPosDiskFace();
        virtual glm::dvec3 operator()(
            const glm::dvec3 &pos) const override;
    } _yPosDiskFace;


    class YNegCircleEdge : public EdgeConstraint
    {
    public:
        YNegCircleEdge();
        virtual glm::dvec3 operator()(
            const glm::dvec3 &pos) const override;
    } _yNegCircleEdge;

    class YPosCircleEdge : public EdgeConstraint
    {
    public:
        YPosCircleEdge();
        virtual glm::dvec3 operator()(
            const glm::dvec3 &pos) const override;
    } _yPosCircleEdge;
};



// IMPLEMENTATION //
inline const FaceConstraint* PipeBoundary::cylinderFace() const
{
    return &_cylinderFace;
}

inline const FaceConstraint* PipeBoundary::yNegDiskFace() const
{
    return &_yNegDiskFace;
}

inline const FaceConstraint* PipeBoundary::yPosDiskFace() const
{
    return &_yPosDiskFace;
}

inline const EdgeConstraint* PipeBoundary::yNegCircleEdge() const
{
    return &_yNegCircleEdge;
}

inline const EdgeConstraint* PipeBoundary::yPosCircleEdge() const
{
    return &_yPosCircleEdge;
}


#endif // GPUMESH_PIPE_BOUNDARY
