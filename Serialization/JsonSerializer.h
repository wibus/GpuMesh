#ifndef GPUMESH_JSONSERIALIZER
#define GPUMESH_JSONSERIALIZER

#include <QJsonValue>

#include "AbstractSerializer.h"
#include "DataStructures/Mesh.h"


class JsonSerializer : public AbstractSerializer
{
public:
    JsonSerializer();
    virtual ~JsonSerializer();

    virtual bool serialize(
            const std::string& fileName,
            const MeshCrew& crew,
            const Mesh& mesh) const override;

protected:
};

#endif // GPUMESH_JSONSERIALIZER
