#ifndef GPUMESH_JSONDESERIALIZER
#define GPUMESH_JSONDESERIALIZER

#include <QJsonValue>

#include "DataStructures/Mesh.h"


class JsonDeserializer
{
public:
    JsonDeserializer();
    virtual ~JsonDeserializer();

    virtual bool deserialize(
            const std::string& fileName,
            Mesh& mesh) const;

protected:
    static MeshVert toVert(const QJsonValue& v);
    static MeshTet  toTet(const QJsonValue& v);
    static MeshPri  toPri(const QJsonValue& v);
    static MeshHex  toHex(const QJsonValue& v);
};

#endif // GPUMESH_JSONDESERIALIZER
