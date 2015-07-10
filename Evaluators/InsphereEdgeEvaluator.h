#ifndef GPUMESH_INSPHEREEDGEEVALUATOR
#define GPUMESH_INSPHEREEDGEEVALUATOR

#include "AbstractEvaluator.h"


class InsphereEdgeEvaluator : public AbstractEvaluator
{
public:
    InsphereEdgeEvaluator();
    virtual ~InsphereEdgeEvaluator();


    virtual double tetrahedronQuality(const Mesh& mesh, const MeshTet& tet) const override;

    virtual double prismQuality(const Mesh& mesh, const MeshPri& pri) const override;

    virtual double hexahedronQuality(const Mesh& mesh, const MeshHex& hex) const override;
};

#endif // GPUMESH_INSPHEREEDGEEVALUATOR
