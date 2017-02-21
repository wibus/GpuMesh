#include "JsonDeserializer.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include "Boundaries/AbstractBoundary.h"
#include "JsonMeshTags.h"


JsonDeserializer::JsonDeserializer()
{
}

JsonDeserializer::~JsonDeserializer()
{

}

bool JsonDeserializer::deserialize(
        const std::string& fileName,
        Mesh& mesh) const
{
    QFile jsonFile(fileName.c_str());
    if(!jsonFile.open(QFile::ReadOnly | QFile::Text))
       return false;

    QJsonDocument doc = QJsonDocument::fromJson(jsonFile.readAll());

    if(!doc.isObject())
        return false;

    QJsonObject meshObj = doc.object();
    mesh.modelName = meshObj[MESH_MODEL_TAG].toString().toStdString();

    std::shared_ptr<AbstractBoundary> bound = boundary(
        meshObj[MESH_BOUND_TAG].toString().toStdString());
    mesh.setBoundary(bound);

    // Vertices
    for(QJsonValue val : meshObj[MESH_VERTS_TAG].toArray())
        mesh.verts.push_back(toVert(val));

    // Topos
    for(QJsonValue val : meshObj[MESH_TOPOS_TAG].toArray())
        mesh.topos.push_back(MeshTopo(bound->constraint(val.toInt())));

    // Tetrahedra
    for(QJsonValue val : meshObj[MESH_TETS_TAG].toArray())
        mesh.tets.push_back(toTet(val));

    // Prisms
    for(QJsonValue val : meshObj[MESH_PRIS_TAG].toArray())
        mesh.pris.push_back(toPri(val));

    // Hexahedra
    for(QJsonValue val : meshObj[MESH_HEXS_TAG].toArray())
        mesh.hexs.push_back(toHex(val));

    return true;
}

MeshVert JsonDeserializer::toVert(const QJsonValue& v)
{
    QJsonArray val = v.toArray();
    MeshVert vert(glm::dvec3(
        val[0].toDouble(),
        val[1].toDouble(),
        val[2].toDouble()));
    return vert;
}

MeshTet JsonDeserializer::toTet(const QJsonValue& v)
{
    QJsonArray val = v.toArray();
    MeshTet elem(
        val[0].toInt(),
        val[1].toInt(),
        val[2].toInt(),
        val[3].toInt());
    return elem;
}

MeshPri JsonDeserializer::toPri(const QJsonValue& v)
{
    QJsonArray val = v.toArray();
    MeshPri elem(
        val[0].toInt(),
        val[1].toInt(),
        val[2].toInt(),
        val[3].toInt(),
        val[4].toInt(),
        val[5].toInt());
    return elem;
}

MeshHex JsonDeserializer::toHex(const QJsonValue& v)
{
    QJsonArray val = v.toArray();
    MeshHex elem(
        val[0].toInt(),
        val[1].toInt(),
        val[2].toInt(),
        val[3].toInt(),
        val[4].toInt(),
        val[5].toInt(),
        val[6].toInt(),
        val[7].toInt());
    return elem;
}
