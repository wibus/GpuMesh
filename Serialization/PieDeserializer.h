#ifndef PIEDESERIALIZER_H
#define PIEDESERIALIZER_H

#ifdef ENBALE_PIRATE

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

#endif // ENABLE_PIRATE

#endif // PIEDESERIALIZER_H
