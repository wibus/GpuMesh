#include "JsonSerializer.h"

#include <fstream>

#include "DataStructures/Mesh.h"
#include "Boundaries/AbstractBoundary.h"
#include "JsonMeshTags.h"

using namespace std;


const string INDENT = "    ";


JsonSerializer::JsonSerializer()
{

}

JsonSerializer::~JsonSerializer()
{

}

bool JsonSerializer::serialize(
        const std::string& fileName,
        const MeshCrew& crew,
        const Mesh& mesh) const
{
    ofstream file;
    file.open(fileName, ios_base::trunc);
    if(!file.is_open())
    {
        return false;
    }

    file << "{\n";
    file << INDENT << '"' << MESH_MODEL_TAG << '"' << ": "
             << '"' << mesh.modelName << '"' << ",\n";

    file << INDENT << '"' << MESH_BOUND_TAG << '"' << ": "
             << '"' << mesh.boundary().name() << '"' << ",\n";

    // Vertices
    file << INDENT << '"' << MESH_VERTS_TAG << '"' << ": [\n";
    for(size_t i=0; i < mesh.verts.size(); ++i)
    {
        const glm::dvec3& p = mesh.verts[i].p;

        file << INDENT << INDENT;
        file << "[" << p.x << ", " << p.y << ", " << p.z << "]";
        if(i != mesh.verts.size() -1) file << ",";
        file << endl;
    }
    file << INDENT << "],\n";

    file << INDENT << '"' << MESH_TOPOS_TAG << '"' << ": [\n";
    for(size_t i=0; i < mesh.topos.size(); ++i)
    {
        const MeshTopo& topo = mesh.topos[i];

        file << INDENT << INDENT << topo.snapToBoundary->id();
        if(i != mesh.topos.size() -1) file << ",";
        file << endl;
    }
    file << INDENT << "],\n";


    // Tetrahedra
    file << INDENT << '"' << MESH_TETS_TAG << '"' << ": [\n";
    for(size_t i=0; i < mesh.tets.size(); ++i)
    {
        const MeshTet& tet = mesh.tets[i];

        file << INDENT << INDENT;
        file << "[" << tet.v[0] << ", " << tet.v[1] << ", "
                    << tet.v[2] << ", " << tet.v[3] << "]";
        if(i != mesh.tets.size() -1) file << ",";
        file << endl;
    }
    file << INDENT << "],\n";


    // Prisms
    file << INDENT << '"' << MESH_PRIS_TAG << '"' << ": [\n";
    for(size_t i=0; i < mesh.pris.size(); ++i)
    {
        const MeshPri& pri = mesh.pris[i];

        file << INDENT << INDENT;
        file << "[" << pri.v[0] << ", " << pri.v[1] << ", "
                    << pri.v[2] << ", " << pri.v[3] << ", "
                    << pri.v[4] << ", " << pri.v[5] << "]";
        if(i != mesh.pris.size() -1) file << ",";
        file << endl;
    }
    file << INDENT << "],\n";


    // Hexahedra
    file << INDENT << '"' << MESH_HEXS_TAG << '"' << ": [\n";
    for(size_t i=0; i < mesh.hexs.size(); ++i)
    {
        const MeshHex& hex = mesh.hexs[i];

        file << INDENT << INDENT;
        file << "[" << hex.v[0] << ", " << hex.v[1] << ", "
                    << hex.v[2] << ", " << hex.v[3] << ", "
                    << hex.v[4] << ", " << hex.v[5] << ", "
                    << hex.v[6] << ", " << hex.v[7] << "]";
        if(i != mesh.hexs.size() -1) file << ",";
        file << endl;
    }
    file << INDENT << "]\n";

    file << "}" << endl;
    file.close();

    return true;
}
