#include "JsonDeserializer.h"

#include <fstream>

#include <CellarWorkbench/Misc/Log.h>

#include "Boundaries/AbstractBoundary.h"
#include "JsonMeshTags.h"

#include "Samplers/ComputedSampler.h"

using namespace std;
using namespace cellar;


JsonDeserializer::JsonDeserializer()
{
}

JsonDeserializer::~JsonDeserializer()
{

}

bool JsonDeserializer::deserialize(
        const std::string& fileName,
        Mesh& mesh,
        const std::shared_ptr<AbstractSampler>& computedSampler) const
{
    ifstream file;
    file.open(fileName);
    if(!file.is_open())
    {
        return false;
    }

    string tag;
    while(readTag(file, tag))
    {
        if(tag == MESH_MODEL_TAG)
        {
            mesh.modelName.clear();
            readString(file, mesh.modelName);
        }
        else if(tag == MESH_BOUND_TAG)
        {
            string boundaryName;
            readString(file, boundaryName);
            mesh.setBoundary(boundary(boundaryName));
        }
        else if(tag == MESH_VERTS_TAG)
        {
            openArray(file);

            vector<double> a;
            while(readArray(file, a))
            {
                mesh.verts.push_back(MeshVert(
                    glm::dvec3(a[0], a[1], a[2])));
                a.clear();
            }

            closeArray(file);
        }
        else if(tag == MESH_TOPOS_TAG)
        {
            vector<int> a;
            if(readArray(file, a))
            {
                for(int i : a)
                {
                    mesh.topos.push_back(MeshTopo(
                        mesh.boundary().constraint(i)));
                }
                a.clear();
            }
        }
        else if(tag == MESH_TETS_TAG)
        {
            openArray(file);

            vector<int> a;
            while(readArray(file, a))
            {
                mesh.tets.push_back(MeshTet(
                    a[0], a[1], a[2], a[3]));
                a.clear();
            }

            closeArray(file);
        }
        else if(tag == MESH_PRIS_TAG)
        {
            openArray(file);

            vector<int> a;
            while(readArray(file, a))
            {
                mesh.pris.push_back(MeshPri(
                    a[0], a[1], a[2], a[3],
                    a[4], a[5]));
                a.clear();
            }

            closeArray(file);
        }
        else if(tag == MESH_HEXS_TAG)
        {
            openArray(file);

            vector<int> a;
            while(readArray(file, a))
            {
                mesh.hexs.push_back(MeshHex(
                    a[0], a[1], a[2], a[3],
                    a[4], a[5], a[6], a[7]));
                a.clear();
            }

            closeArray(file);
        }
        else
        {
            getLog().postMessage(new Message('E', false,
                "Unknown tag found while reading json mesh: " + tag,
                "JsonDeserializer"));
            file.close();
            return false;
        }

        tag.clear();
    }

    file.close();


    static_cast<ComputedSampler*>(computedSampler.get())
        ->setComputedMetrics(mesh,vector<MeshMetric>(mesh.verts.size()));

    return true;
}

bool JsonDeserializer::readTag(std::istream& is, std::string& str) const
{
    char c = '\0';
    while(c != '"' && !is.eof())
    {
        is.get(c);
    }

    if(is.eof())
        return false;

    is.get(c);
    while(c != '"' && !is.eof())
    {
        str.push_back(c);
        is.get(c);
    }

    if(is.eof())
        return false;

    return true;
}

bool JsonDeserializer::readString(std::istream& is, std::string& str) const
{
    char c = '\0';
    while(c != '"' && !is.eof())
    {
        is.get(c);
    }

    if(is.eof())
        return false;

    is.get(c);
    while(c != '"' && !is.eof())
    {
        str.push_back(c);
        is.get(c);
    }

    if(is.eof())
        return false;

    return true;
}

bool JsonDeserializer::openArray(std::istream &is) const
{
    char c = '\0';
    while(c != '[' && !is.eof())
    {
        is.get(c);
    }

    if(is.eof())
        return false;

    return true;
}

bool JsonDeserializer::closeArray(std::istream &is) const
{
    char c = '\0';
    while(c != ']' && c != ',' && !is.eof())
    {
        is.get(c);
    }

    if(is.eof())
        return false;

    return true;
}

template<typename T>
bool JsonDeserializer::readArray(std::istream &is, std::vector<T>& a) const
{
    char c = '\0';
    while(c != '[' && c!= ']' && !is.eof())
    {
        is.get(c);
    }

    if(c == ']' || is.eof())
        return false;

    T v;
    while(is >> v)
    {
        a.push_back(v);

        is.get(c);
        while(c != ',' && c != ']')
            is.get(c);

        if(c == ']')
            break;
    }

    is.clear();

    return true;
}
