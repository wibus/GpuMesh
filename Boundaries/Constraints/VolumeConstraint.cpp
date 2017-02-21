#include "VolumeConstraint.h"

#include "FaceConstraint.h"


VolumeConstraint::VolumeConstraint() :
    AbstractConstraint(0, 3)
{

}

const AbstractConstraint* VolumeConstraint::subconstraint(int id) const
{
    if(id == this->id())
        return this;

    const AbstractConstraint* constraint = nullptr;
    for(const FaceConstraint* f : _faces)
    {
        constraint = f->subconstraint(id);
        if(constraint != nullptr)
            return constraint;
    }

    return nullptr;
}

void VolumeConstraint::addFace(FaceConstraint* face)
{
    if(!isBoundedBy(face))
    {
        _faces.push_back(face);

        face->addVolume(this);
    }
}

bool VolumeConstraint::isBoundedBy(const FaceConstraint* face) const
{
    for(const FaceConstraint* s : _faces)
        if(s == face)
            return true;

    return false;
}

glm::dvec3 VolumeConstraint::operator()(const glm::dvec3& pos) const
{
    return pos;
}

const AbstractConstraint* VolumeConstraint::split(const AbstractConstraint* c) const
{
    return this;
}

const AbstractConstraint* VolumeConstraint::merge(const AbstractConstraint* c) const
{
    return c;
}
