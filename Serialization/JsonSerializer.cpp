#include "JsonSerializer.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include "DataStructures/Mesh.h"
#include "JsonMeshTags.h"


JsonSerializer::JsonSerializer()
{

}

JsonSerializer::~JsonSerializer()
{

}

bool JsonSerializer::serialize(
        const std::string& fileName,
        const Mesh& mesh) const
{
    QJsonObject meshObj;
    meshObj.insert(MESH_MODEL_TAG, mesh.modelName.c_str());

    // Vertices
    QJsonArray vertArray;
    for(const MeshVert& vert : mesh.verts)
        vertArray.append(toJson(vert.p));
    meshObj.insert(MESH_VERTS_TAG, vertArray);


    // Tetrahedra
    QJsonArray tetArray;
    for(const MeshTet& elem : mesh.tets)
        tetArray.append(toJson(elem));
    meshObj.insert(MESH_TETS_TAG, tetArray);


    // Prisms
    QJsonArray priArray;
    for(const MeshPri& elem : mesh.pris)
        priArray.append(toJson(elem));
    meshObj.insert(MESH_PRIS_TAG, priArray);


    // Hexahedra
    QJsonArray hexArray;
    for(const MeshHex& elem : mesh.hexs)
        hexArray.append(toJson(elem));
    meshObj.insert(MESH_HEXS_TAG, hexArray);


    QJsonDocument doc(meshObj);
    QFile jsonFile(fileName.c_str());
    if(!jsonFile.open(QFile::WriteOnly))
        return false;

    jsonFile.write(doc.toJson(QJsonDocument::Compact));
    return true;
}

QJsonValue JsonSerializer::toJson(const MeshVert& v)
{
    QJsonArray val;
    val.append(v.p.x);
    val.append(v.p.y);
    val.append(v.p.z);
    return val;
}

QJsonValue JsonSerializer::toJson(const MeshTet& e)
{
    QJsonArray val;
    val.append((int) e.v[0]);
    val.append((int) e.v[1]);
    val.append((int) e.v[2]);
    val.append((int) e.v[3]);
    return val;
}

QJsonValue JsonSerializer::toJson(const MeshPri& e)
{
    QJsonArray val;
    val.append((int) e.v[0]);
    val.append((int) e.v[1]);
    val.append((int) e.v[2]);
    val.append((int) e.v[3]);
    val.append((int) e.v[4]);
    val.append((int) e.v[5]);
    return val;
}

QJsonValue JsonSerializer::toJson(const MeshHex& e)
{
    QJsonArray val;
    val.append((int) e.v[0]);
    val.append((int) e.v[1]);
    val.append((int) e.v[2]);
    val.append((int) e.v[3]);
    val.append((int) e.v[4]);
    val.append((int) e.v[5]);
    val.append((int) e.v[6]);
    val.append((int) e.v[7]);
    return val;
}
