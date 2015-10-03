#ifndef GPUMESH_KDTREEDISCRETIZER
#define GPUMESH_KDTREEDISCRETIZER

#include "AbstractDiscretizer.h"


class KdTreeDiscretizer : public AbstractDiscretizer
{
public:
    KdTreeDiscretizer();
    virtual ~KdTreeDiscretizer();

    virtual std::shared_ptr<Mesh> gridMesh() const override;

    virtual void discretize(
            const Mesh& mesh,
            const glm::ivec3& gridSize) override;

private:
    std::shared_ptr<Mesh> _gridMesh;
};

#endif // GPUMESH_KDTREEDISCRETIZER
