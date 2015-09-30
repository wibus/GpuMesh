#include "StlSerializer.h"

#include <iostream>

#include "DataStructures/Mesh.h"
#include "Evaluators/AbstractEvaluator.h"
#include "UserInterface/Dialogs/StlSerializerDialog.h"


using namespace std;


StlSerializer::StlSerializer()
{

}

StlSerializer::~StlSerializer()
{

}

bool StlSerializer::serialize(
        const std::string& fileName,
        const AbstractEvaluator& evaluator,
        const Mesh& mesh) const
{
    StlSerializerDialog dialog;
    int returnCode = dialog.exec();
    if(returnCode != QDialog::Accepted)
        return false;

    auto headerFunc = &StlSerializer::headerAscii;
    auto footerFunc = &StlSerializer::footerAscii;
    auto printFunc = &StlSerializer::printAscii;
    bool computeQuality = false;
    bool binaryMode = false;
    if(dialog.binaryFormat())
    {
        binaryMode = true;
        headerFunc = &StlSerializer::headerBin;
        footerFunc = &StlSerializer::footerBin;
        if(dialog.embedQuality())
        {
            computeQuality = true;
            printFunc = &StlSerializer::printColorBin;
        }
        else
        {
            printFunc = &StlSerializer::printFlatBin;
        }
    }


    // Open file
    auto openMode = ios_base::out | ios_base::trunc;
    if(binaryMode) openMode |= ios_base::binary;
    ofstream stlFile(fileName, openMode);
    if(!stlFile.is_open())
        return false;

    headerFunc(stlFile, mesh);

    size_t tetCount = mesh.tets.size();
    for(size_t e=0; e < tetCount; ++e)
    {
        const MeshTet& elem = mesh.tets[e];
        for(size_t t=0; t < MeshTet::TRI_COUNT; ++t)
        {
            glm::vec3 v[] = {
                glm::vec3(mesh.verts[elem.v[MeshTet::tris[t][0]]].p),
                glm::vec3(mesh.verts[elem.v[MeshTet::tris[t][1]]].p),
                glm::vec3(mesh.verts[elem.v[MeshTet::tris[t][2]]].p)};
            double qual = computeQuality ? evaluator.tetQuality(mesh, elem) : 0;
            printFunc(stlFile, v, qual);
        }
    }

    size_t priCount = mesh.pris.size();
    for(size_t e=0; e < priCount; ++e)
    {
        const MeshPri& elem = mesh.pris[e];
        for(size_t t=0; t < MeshPri::TRI_COUNT; ++t)
        {
            glm::vec3 v[] = {
                glm::vec3(mesh.verts[elem.v[MeshPri::tris[t][0]]].p),
                glm::vec3(mesh.verts[elem.v[MeshPri::tris[t][1]]].p),
                glm::vec3(mesh.verts[elem.v[MeshPri::tris[t][2]]].p)};
            double qual = computeQuality ? evaluator.priQuality(mesh, elem) : 0;
            printFunc(stlFile, v, qual);
        }
    }

    size_t hexCount = mesh.hexs.size();
    for(size_t e=0; e < hexCount; ++e)
    {
        const MeshHex& elem = mesh.hexs[e];
        for(size_t t=0; t < MeshHex::TRI_COUNT; ++t)
        {
            glm::vec3 v[] = {
                glm::vec3(mesh.verts[elem.v[MeshHex::tris[t][0]]].p),
                glm::vec3(mesh.verts[elem.v[MeshHex::tris[t][1]]].p),
                glm::vec3(mesh.verts[elem.v[MeshHex::tris[t][2]]].p)};
            double qual = computeQuality ? evaluator.hexQuality(mesh, elem) : 0;
            printFunc(stlFile, v, qual);
        }
    }

    footerFunc(stlFile, mesh);

    stlFile.close();
    return true;
}

void StlSerializer::headerAscii(ofstream& stream, const Mesh& mesh)
{
    stream << "solid " << mesh.modelName << endl;
}

void StlSerializer::headerBin(ofstream& stream, const Mesh& mesh)
{
    char header[80];
    memset(header, 0, sizeof(header));
    stream.write(header, sizeof(header));

    unsigned int triCount =
        mesh.tets.size() * MeshTet::TRI_COUNT +
        mesh.pris.size() * MeshPri::TRI_COUNT +
        mesh.hexs.size() * MeshHex::TRI_COUNT;

    stream.write((char*)&triCount, sizeof(triCount));
}

void StlSerializer::footerAscii(ofstream& stream, const Mesh& mesh)
{
    stream << "endsolid" << endl;
}

void StlSerializer::footerBin(ofstream& stream, const Mesh& mesh)
{
    // nothing to be writen
}

void StlSerializer::printAscii(ofstream& stream, const glm::vec3 v[], double qual)
{
    glm::vec3 n(glm::cross(v[1]-v[0], v[2] - v[1]));
    n = glm::normalize(n);

    stream << "  facet normal " << n.x << " " << n.y << " " << n.z << endl;
    stream << "    outer loop" << endl;
    stream << "      vertex " << v[0].x << " " << v[0].y << " " << v[0].z << endl;
    stream << "      vertex " << v[1].x << " " << v[1].y << " " << v[1].z << endl;
    stream << "      vertex " << v[2].x << " " << v[2].y << " " << v[2].z << endl;
    stream << "    endloop" << endl;
    stream << "  endfacet" << endl;
}

void StlSerializer::printFlatBin(ofstream& stream, const glm::vec3 v[], double qual)
{
    glm::vec3 n(glm::cross(v[1]-v[0], v[2] - v[1]));
    n = glm::normalize(n);

    unsigned short attCount = 0;

    stream.write((char*)&n, sizeof(glm::vec3));
    stream.write((char*)&v[0], 3*sizeof(glm::vec3));
    stream.write((char*)&attCount, sizeof(attCount));
}

void StlSerializer::printColorBin(ofstream& stream, const glm::vec3 v[], double qual)
{
    glm::vec3 n(glm::cross(v[1]-v[0], v[2] - v[1]));
    n = glm::normalize(n);

    glm::vec3 grad(
        // Red
        1.0 - glm::smoothstep(0.25, 0.5, qual),
        // Green
        glm::smoothstep(0.0, 0.25, qual) - glm::smoothstep(0.75, 1.0, qual),
        // Blue
        glm::smoothstep(0.5, 0.75, qual));

    unsigned int r = grad.r * 31;
    unsigned int g = grad.g * 31;
    unsigned int b = grad.b * 31;

    const unsigned int rShift = 10;
    const unsigned int gShift = 5;
    const unsigned int bShift = 0;

    const unsigned int vMask = 1 << 15;

    unsigned short attCount =
            ((r << rShift)) |
            ((g << gShift)) |
            ((b << bShift)) |
            (qual <= 0.0 ? vMask : 0);

    stream.write((char*)&n, sizeof(glm::vec3));
    stream.write((char*)&v[0], 3*sizeof(glm::vec3));
    stream.write((char*)&attCount, sizeof(attCount));
}
