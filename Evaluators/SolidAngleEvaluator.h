#ifndef GPUMESH_SOLIDANGLEEVALUATOR
#define GPUMESH_SOLIDANGLEEVALUATOR

#include "AbstractEvaluator.h"


class SolidAngleEvaluator : public AbstractEvaluator
{
public:
    SolidAngleEvaluator();
    virtual ~SolidAngleEvaluator();

    virtual double solidAngle(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c) const;

    virtual double tetrahedronQuality(const glm::dvec3 verts[]) const override;

    virtual double prismQuality(const glm::dvec3 verts[]) const override;

    virtual double hexahedronQuality(const glm::dvec3 verts[]) const override;
};

#endif // GPUMESH_SOLIDANGLEEVALUATOR
