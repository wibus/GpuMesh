#include "CgnsDeserializer.h"

#include <CellarWorkbench/Misc/Log.h>

#include "cgnslib.h"

#include "Boundaries/AbstractBoundary.h"
#include "Boundaries/Constraints/VertexConstraint.h"


using namespace std;
using namespace cellar;

class CgnsBoundary : public AbstractBoundary
{
public:
    CgnsBoundary() :
        AbstractBoundary("CGNS Boundary"),
        vertex(-1, glm::dvec3(0, 0, 0))
    {

    }

    virtual ~CgnsBoundary()
    {

    }

    virtual bool unitTest() const override
    {
        return true;
    }

    const AbstractConstraint* fixedConstrait() const
    {
        return &vertex;
    }

private:
    VertexConstraint vertex;
};


CgnsDeserializer::CgnsDeserializer()
{

}

CgnsDeserializer::~CgnsDeserializer()
{

}

bool CgnsDeserializer::deserialize(
        const std::string& fileName,
        Mesh& mesh) const
{
    string indent = "";

    shared_ptr<CgnsBoundary> cgnsBoundary(
        make_shared<CgnsBoundary>());
    mesh.setBoundary(cgnsBoundary);

    int fn = -1;
    if(cg_open(fileName.c_str(), CG_MODE_READ, &fn))
    {
        getLog().postMessage(new Message('E', false,
            indent + "Could not open CGNS file", "CgnsDeserializer"));
        return false;
    }

    int nbases = -1;
    if(cg_nbases(fn, &nbases))
    {
        getLog().postMessage(new Message('E', false,
            indent + "Could not read CGNS file's number of bases", "CgnsDeserializer"));
        cg_close(fn);
        return false;
    }

    for(int b=1; b <= nbases; ++b)
    {
        string indent = "   ";

        char baseName[256];
        int cell_dim = -1,
            phys_dim = -1;

        if(cg_base_read(fn, b, baseName, &cell_dim, &phys_dim))
        {
            getLog().postMessage(new Message('E', false,
                indent + "Could not read CGNS base", "CgnsDeserializer"));
            cg_close(fn);
            return false;
        }

        getLog().postMessage(new Message('D', false,
            indent + "Reading CGNS base: " + std::string(baseName), "CgnsDeserializer"));

        if(phys_dim != 3)
        {
            getLog().postMessage(new Message('W', false,
                indent + "CGNS base is not 3D. This base will be ignored", "CgnsDeserializer"));
            continue;
        }

        if(cell_dim != 3)
        {
            getLog().postMessage(new Message('W', false,
                indent + "CGNS base does not contain volume elements. This base will be ignored", "CgnsDeserializer"));
            continue;
        }

        int nzones = -1;
        if(cg_nzones(fn, b, &nzones))
        {
            getLog().postMessage(new Message('E', false,
                indent + "Could not read CGNS base's number of zones", "CgnsDeserializer"));
            cg_close(fn);
            return false;
        }

        for(int z=1; z <= nzones; ++z)
        {
            string indent = "      ";

            char zoneName[256];
            cgsize_t zoneSize[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
            if(cg_zone_read(fn, b, z, zoneName, zoneSize))
            {
                getLog().postMessage(new Message('E', false,
                    indent + "Could not read CGNS zone", "CgnsDeserializer"));
                cg_close(fn);
                return false;
            }

            cgsize_t vMin = 1, vMax = zoneSize[0], nVert = zoneSize[0];
            cgsize_t cMin = 1, cMax = zoneSize[1], nCell = zoneSize[1];
            getLog().postMessage(new Message('D', false,
                indent + "Reading CGNS zone: " + std::string(zoneName) +
                " (nv=" + to_string(zoneSize[0]) +
                ", nc=" + to_string(zoneSize[1]) +
                ", nbv=" + to_string(zoneSize[2]) +
                ")" , "CgnsDeserializer"));


            // Zone type
            ZoneType_t zoneType;
            if(cg_zone_type(fn, b, z, &zoneType))
            {
                getLog().postMessage(new Message('E', false,
                    indent + "Could not read CGNS zone's type", "CgnsDeserializer"));
                cg_close(fn);
                return false;
            }

            if(zoneType != Unstructured)
            {
                getLog().postMessage(new Message('W', false,
                    indent + "CGNS file contains a structured zone. This zone will be ignored", "CgnsDeserializer"));
                continue;
            }


            // Coords
            int ncoords = -1;
            if(cg_ncoords(fn, b, z, &ncoords))
            {
                getLog().postMessage(new Message('E', false,
                    indent + "Could not read CGNS zone's number of coords", "CgnsDeserializer"));
                cg_close(fn);
                return false;
            }

            if(ncoords != 3)
            {
                getLog().postMessage(new Message('W', false,
                    indent + "CGNS zone's coords are not 3D. This zone will be ignored", "CgnsDeserializer"));
                continue;
            }


            // Resize mesh's vertices vector
            size_t baseVertIdx = mesh.verts.size();
            size_t newVertSize = baseVertIdx + nVert;
            mesh.verts.resize(newVertSize);
            mesh.topos.resize(newVertSize);


            // Read coordinates
            for(int c=1; c <= ncoords; ++c)
            {
                string indent = "         ";

                // Data type
                char coordName[256];
                DataType_t coordDataType;
                if(cg_coord_info(fn, b, z, c, &coordDataType, coordName))
                {
                    getLog().postMessage(new Message('E', false,
                        indent + "Could not read CGNS coord's info", "CgnsDeserializer"));
                    cg_close(fn);
                    return false;
                }

                getLog().postMessage(new Message('D', false,
                    indent + "Reading CGNS coord: " + std::string(coordName) +
                    " (type=" + DataTypeName[coordDataType] + ")", "CgnsDeserializer"));


                if(coordDataType == RealSingle)
                {
                    float* coord = new float[vMax];
                    if(cg_coord_read(fn, b, z, coordName, RealSingle, &vMin, &vMax, coord))
                    {
                        getLog().postMessage(new Message('E', false,
                            indent + "Could not read CGNS coord", "CgnsDeserializer"));
                        cg_close(fn);
                        return false;
                    }

                    for(size_t vi = baseVertIdx, ci=0; vi < newVertSize; ++vi, ++ci)
                    {
                        mesh.verts[vi].p[c-1] = coord[ci];
                    }
                }
                else if(coordDataType == RealDouble)
                {
                    double* coord = new double[vMax];
                    if(cg_coord_read(fn, b, z, coordName, RealDouble, &vMin, &vMax, coord))
                    {
                        getLog().postMessage(new Message('E', false,
                            indent + "Could not read CGNS coord", "CgnsDeserializer"));
                        cg_close(fn);
                        return false;
                    }

                    for(size_t vi = baseVertIdx, ci=0; vi < newVertSize; ++vi, ++ci)
                    {
                        mesh.verts[vi].p[c-1] = coord[ci];
                    }
                }
                else
                {
                    getLog().postMessage(new Message('E', false,
                        indent + "CGNS coord's data type is not supported", "CgnsDeserializer"));
                    break;
                }
            }


            // Sections
            int nsections = -1;
            if(cg_nsections(fn, b, z, &nsections))
            {
                getLog().postMessage(new Message('E', false,
                    indent + "Could not read CGNS zone's number of sections", "CgnsDeserializer"));
                cg_close(fn);
                return false;
            }

            getLog().postMessage(new Message('D', false,
                indent + string(zoneName) + " has " +
                to_string(nsections) + " sections",
                "CgnsDeserializer"));


            // Read sections
            for(int s=1; s <= nsections; ++s)
            {
                string indent = "         ";

                char sectionName[256];
                ElementType_t sectionElemType;
                cgsize_t sStart = -1, sEnd = -1;
                int nBoundary = -1;
                int parentFlag = -1;

                if(cg_section_read(fn, b, z, s,
                    sectionName, &sectionElemType,
                    &sStart, &sEnd, &nBoundary, &parentFlag))
                {
                    getLog().postMessage(new Message('E', false,
                        indent + "Could not read CGNS section", "CgnsDeserializer"));
                    cg_close(fn);
                    return false;
                }

                cgsize_t nelems = sEnd - sStart + 1;
                getLog().postMessage(new Message('D', false,
                    indent + "Reading CGNS section: " + std::string(sectionName) +
                    " (elementType=" + ElementTypeName[sectionElemType] +
                    ", nbElems=" + to_string(nelems) +
                    ")", "CgnsDeserializer"));


                // Element data size
                cgsize_t eSize = -1;
                if(cg_ElementPartialSize(fn, b, z, s, sStart, sEnd, &eSize))
                {
                    getLog().postMessage(new Message('E', false,
                        indent + "Could not read CGNS section's size", "CgnsDeserializer"));
                    cg_close(fn);
                    return false;
                }

                cgsize_t* parentData = nullptr;
                cgsize_t* elems = new cgsize_t[eSize];
                if(cg_elements_partial_read(fn, b, z, s, sStart, sEnd, elems, parentData))
                {
                    getLog().postMessage(new Message('E', false,
                        indent + "Could not read CGNS section's elements", "CgnsDeserializer"));
                    cg_close(fn);
                    return false;
                }

                if(sectionElemType == TRI_3 ||
                   sectionElemType == QUAD_4)
                {
                    for(size_t bi = 0; bi < eSize; ++bi)
                    {
                        mesh.topos[elems[bi]-1] = MeshTopo(
                            cgnsBoundary->fixedConstrait());
                    }
                }
                else if(sectionElemType == TETRA_4)
                {

                    size_t baseElemIdx = mesh.tets.size();
                    size_t newElemSize = baseElemIdx + nelems;
                    mesh.tets.resize(newElemSize);

                    for(size_t ti = baseElemIdx, ei=sStart-1; ti < newElemSize; ++ti, ++ei)
                    {
                        size_t eb = ei*4;
                        mesh.tets[ti] = MeshTet(
                            elems[eb + 0]-1, elems[eb + 1]-1,
                            elems[eb + 2]-1, elems[eb + 3]-1);
                    }
                }
                else if(sectionElemType == PYRA_5)
                {
                    getLog().postMessage(new Message('W', false,
                        indent + "CGNS PYRA_5 is not yet supported. This section will be ignored", "CgnsDeserializer"));
                    continue;
                }
                else if(sectionElemType == PENTA_6)
                {
                    size_t baseElemIdx = mesh.pris.size();
                    size_t newElemSize = baseElemIdx + nelems;
                    mesh.pris.resize(newElemSize);

                    for(size_t pi = baseElemIdx, ei=sStart-1; pi < newElemSize; ++pi, ++ei)
                    {
                        size_t eb = ei*6;
                        mesh.pris[pi] = MeshPri(
                            elems[eb + 0]-1, elems[eb + 1]-1, elems[eb + 2]-1,
                            elems[eb + 3]-1, elems[eb + 4]-1, elems[eb + 5]-1);
                    }
                }
                else if(sectionElemType == HEXA_8)
                {
                    size_t baseElemIdx = mesh.hexs.size();
                    size_t newElemSize = baseElemIdx + nelems;
                    mesh.hexs.resize(newElemSize);

                    for(size_t hi = baseElemIdx, ei=sStart-1; hi < newElemSize; ++hi, ++ei)
                    {
                        size_t eb = ei*8;
                        mesh.hexs[hi] = MeshHex(
                            elems[eb + 0]-1, elems[eb + 1]-1, elems[eb + 2]-1, elems[eb + 3]-1,
                            elems[eb + 4]-1, elems[eb + 5]-1, elems[eb + 6]-1, elems[eb + 7]-1);
                    }
                }
                else if(sectionElemType == MIXED)
                {
                    size_t eb = 0;
                    bool pyrFound = false;
                    bool otherFound = false;
                    for(cgsize_t ei=0; ei < nelems; ++ei)
                    {
                        ElementType_t et = ElementType_t(elems[eb]);
                        ++eb;

                        switch(et)
                        {
                        case TRI_3 :
                            mesh.topos[elems[eb + 0]-1] = MeshTopo(
                                cgnsBoundary->fixedConstrait());
                            mesh.topos[elems[eb + 1]-1] = MeshTopo(
                                cgnsBoundary->fixedConstrait());
                            mesh.topos[elems[eb + 2]-1] = MeshTopo(
                                cgnsBoundary->fixedConstrait());
                            break;

                        case QUAD_4 :
                            mesh.topos[elems[eb + 0]-1] = MeshTopo(
                                cgnsBoundary->fixedConstrait());
                            mesh.topos[elems[eb + 1]-1] = MeshTopo(
                                cgnsBoundary->fixedConstrait());
                            mesh.topos[elems[eb + 2]-1] = MeshTopo(
                                cgnsBoundary->fixedConstrait());
                            mesh.topos[elems[eb + 3]-1] = MeshTopo(
                                cgnsBoundary->fixedConstrait());
                            break;

                        case TETRA_4 :
                            mesh.tets.push_back(MeshTet(
                                elems[eb + 0]-1, elems[eb + 1]-1,
                                elems[eb + 2]-1, elems[eb + 3]-1));
                            break;

                        case PYRA_5 :
                            pyrFound = true;
                            break;

                        case PENTA_6 :
                            mesh.pris.push_back(MeshPri(
                                elems[eb + 0]-1, elems[eb + 1]-1, elems[eb + 2]-1,
                                elems[eb + 3]-1, elems[eb + 4]-1, elems[eb + 5]-1));
                            break;

                        case HEXA_8 :
                            mesh.hexs.push_back(MeshHex(
                                elems[eb + 0]-1, elems[eb + 1]-1, elems[eb + 2]-1, elems[eb + 3]-1,
                                elems[eb + 4]-1, elems[eb + 5]-1, elems[eb + 6]-1, elems[eb + 7]-1));
                            break;

                        default:
                            otherFound = true;
                            break;
                        }

                        int npe = -1;
                        cg_npe(et, &npe);
                        eb += npe;
                    }

                    if(pyrFound)
                    {
                        getLog().postMessage(new Message('W', false,
                            indent + "Some pyramids were found, but ignored", "CgnsDeserializer"));
                    }
                    if(otherFound)
                    {
                        getLog().postMessage(new Message('W', false,
                            indent + "Unsupported element types were found, but ignored", "CgnsDeserializer"));
                    }
                }
                else
                {
                    getLog().postMessage(new Message('W', false,
                        indent + "CGNS section's element type is not supported. This section will be ignored", "CgnsDeserializer"));
                    continue;
                }


                /*
                // Read boundaries
                int nbocos = -1;
                if(cg_nbocos(fn, b, z, &nbocos))
                {
                    getLog().postMessage(new Message('E', false,
                        indent + "Could not read CGNS section's number of boundary conditions", "CgnsDeserializer"));
                    cg_close(fn);
                    return false;
                }

                for(int bc=1; bc <= nbocos; ++bc)
                {
                    GridLocation_t gridLoc;
                    if(cg_boco_gridlocation_read(fn, b, z, bc, &gridLoc))
                    {
                        getLog().postMessage(new Message('E', false,
                            indent + "Could not read CGNS boundary condition's location", "CgnsDeserializer"));
                        cg_close(fn);
                        return false;
                    }

                    getLog().postMessage(new Message('D', false,
                        indent + "Reading CGNS boundary condition (" +
                        "loc=" + GridLocationName[gridLoc] + ")", "CgnsDeserializer"));

                    if(gridLoc != Vertex)
                    {
                        getLog().postMessage(new Message('E', false,
                            indent + "Unsupported boundary condition location: " + GridLocationName[gridLoc], "CgnsDeserializer"));
                        continue;
                    }
                }
                */
            }
        }
    }


    cg_close(fn);

    return true;
}
