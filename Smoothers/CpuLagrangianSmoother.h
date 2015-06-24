#ifndef GPUMESH_CPULANGRANGIANSMOOTHER
#define GPUMESH_CPULANGRANGIANSMOOTHER


#include "AbstractSmoother.h"


class CpuLangrangianSmoother : public AbstractSmoother
{
public:
    CpuLangrangianSmoother(Mesh& mesh, double moveFactor, double gainThreshold);
    virtual ~CpuLangrangianSmoother();

    virtual void smoothMesh() override;
};

#endif // GPUMESH_CPULANGRANGIANSMOOTHER
