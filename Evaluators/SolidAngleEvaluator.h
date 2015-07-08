#ifndef GPUMESH_SOLIDANGLEEVALUATOR
#define GPUMESH_SOLIDANGLEEVALUATOR

#include "AbstractEvaluator.h"


class SolidAngleEvaluator : public AbstractEvaluator
{
public:
    SolidAngleEvaluator();
    virtual ~SolidAngleEvaluator();


    virtual double tetrahedronQuality(const Mesh& mesh, const MeshTet& tet) const override;

    virtual double prismQuality(const Mesh& mesh, const MeshPri& pri) const override;

    virtual double hexahedronQuality(const Mesh& mesh, const MeshHex& hex) const override;

protected:
    virtual double solidAngle(
            const glm::dvec3& a,
            const glm::dvec3& b,
            const glm::dvec3& c) const;
};

#endif // GPUMESH_SOLIDANGLEEVALUATOR
