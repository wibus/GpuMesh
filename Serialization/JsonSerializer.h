#ifndef GPUMESH_JSONSERIALIZER
#define GPUMESH_JSONSERIALIZER

#include <QJsonValue>

#include "DataStructures/Mesh.h"


class JsonSerializer
{
public:
    JsonSerializer();
    virtual ~JsonSerializer();

    virtual bool serialize(
            const std::string& fileName,
            const Mesh& mesh) const;

protected:
    static QJsonValue toJson(const MeshVert& v);
    static QJsonValue toJson(const MeshTet& e);
    static QJsonValue toJson(const MeshPri& e);
    static QJsonValue toJson(const MeshHex& e);
};

#endif // GPUMESH_JSONSERIALIZER
