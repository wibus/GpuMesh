#ifndef CGNSDESERIALIZER_H
#define CGNSDESERIALIZER_H

#include "AbstractDeserializer.h"
#include "DataStructures/Mesh.h"


class CgnsDeserializer : public AbstractDeserializer
{
public:
    CgnsDeserializer();
    virtual ~CgnsDeserializer();

    virtual bool deserialize(
            const std::string& fileName,
            Mesh& mesh) const override;
};

#endif // CGNSDESERIALIZER_H
