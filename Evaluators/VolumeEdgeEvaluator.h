#ifndef GPUMESH_VOLUMEEDGEEVALUATOR
#define GPUMESH_VOLUMEEDGEEVALUATOR

#include "AbstractEvaluator.h"


// Will not give good results for polyhedral element other than the tetrahedron:
// Using the ratio between signed volume and the sum of edges sqared cubed gives
// quality measure higher than one for non regular elements. This _looks_ like a
// consequence of how the polyhedral element is subdivided into sub tetrahedra.
// S. H. Lo applys this 'evaluation method' only for simplicial elements.
// ref : Finite Element Mesh Generation, S. H. Lo, p. 337, sec 6.2.4
class VolumeEdgeEvaluator : public AbstractEvaluator
{
public:
    VolumeEdgeEvaluator();
    virtual ~VolumeEdgeEvaluator();

    virtual double tetQuality(
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[]) const override;

    // !Gives wrong results!
    virtual double priQuality(
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[]) const override;

    // !Gives wrong results!
    virtual double hexQuality(
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[]) const override;
};

#endif // GPUMESH_VOLUMEEDGEEVALUATOR
