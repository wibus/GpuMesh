#ifndef GPUMESH_BOUNDARY_FREE
#define GPUMESH_BOUNDARY_FREE

#include "AbstractBoundary.h"


class BoundaryFree : public AbstractBoundary
{
public:
    BoundaryFree();
    virtual ~BoundaryFree();


    virtual bool unitTest() const override;
};

#endif // GPUMESH_BOUNDARY_FREE
