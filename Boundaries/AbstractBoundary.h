#ifndef GPUMESH_ABSTRACT_BOUNDARY
#define GPUMESH_ABSTRACT_BOUNDARY

#include <vector>
#include <string>

#include "Constraints/VertexConstraint.h"
#include "Constraints/EdgeConstraint.h"
#include "Constraints/FaceConstraint.h"
#include "Constraints/VolumeConstraint.h"


typedef void (*ModelBoundsCudaFct)(void);


class AbstractBoundary
{
protected:
    AbstractBoundary(
            const std::string& name,
            const std::string& shaderName,
            ModelBoundsCudaFct cudaBoundary);

public:
    virtual ~AbstractBoundary();



    virtual bool unitTest() const = 0;


    void installCudaPlugIn() const;

    const std::string& name() const;

    const std::string& shaderName() const;

    const VolumeConstraint* volume() const;

    int supportDimension(
        const AbstractConstraint* c1,
        const AbstractConstraint* c2) const;

    virtual const AbstractConstraint* split(
        const AbstractConstraint* c1,
        const AbstractConstraint* c2) const;

    virtual const AbstractConstraint* merge(
        const AbstractConstraint* c1,
        const AbstractConstraint* c2) const;

    static const AbstractConstraint* INVALID_OPERATION;

protected:
    VolumeConstraint* volume();

private:
    std::string _name;
    std::string _shaderName;
    ModelBoundsCudaFct _cudaBoundary;

    VolumeConstraint _volume;
};



// IMPLEMENTATION //
inline void AbstractBoundary::installCudaPlugIn() const
{
    (*_cudaBoundary)();
}

inline const std::string& AbstractBoundary::name() const
{
    return _name;
}

inline const std::string& AbstractBoundary::shaderName() const
{
    return _shaderName;
}

inline const VolumeConstraint* AbstractBoundary::volume() const
{
    return &_volume;
}

inline VolumeConstraint* AbstractBoundary::volume()
{
    return &_volume;
}


#endif // GPUMESH_ABSTRACT_BOUNDARY
