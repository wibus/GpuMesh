#ifndef GPUMESH_KDTREEDISCRETIZER
#define GPUMESH_KDTREEDISCRETIZER

#include <vector>

#include "AbstractDiscretizer.h"

struct KdNode;


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
    void build(
            KdNode* node,
            int height,
            const Mesh& mesh,
            const glm::dvec3& minBox,
            const glm::dvec3& maxBox,
            const std::vector<uint>& xSort,
            const std::vector<uint>& ySort,
            const std::vector<uint>& zSort);
    void meshTree(KdNode* node, Mesh& mesh);

    std::shared_ptr<Mesh> _gridMesh;
};

#endif // GPUMESH_KDTREEDISCRETIZER
