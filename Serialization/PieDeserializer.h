#ifndef PIEDESERIALIZER_H
#define PIEDESERIALIZER_H

#include "AbstractDeserializer.h"
#include "DataStructures/Mesh.h"


class PieDeserializer : public AbstractDeserializer
{
public:
    PieDeserializer();
    virtual ~PieDeserializer();

    virtual bool deserialize(
            const std::string& fileName,
            Mesh& mesh,
            std::vector<MeshMetric>& metrics) const override;
};

#endif // PIEDESERIALIZER_H
