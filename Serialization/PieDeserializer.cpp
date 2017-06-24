#include "PieDeserializer.h"

#include <CellarWorkbench/Misc/Log.h>

#include <Pir/pifich.h>
#include <Pir/pmzone.h>
#include <Pir/ptintgeo.h>
#include <Pir/PS_SOLUTION.h>
#include <Pir/PS_VARIABLE.h>
#include <Pir/pmelement.h>

#include "Boundaries/AbstractBoundary.h"
#include "Boundaries/Constraints/VertexConstraint.h"


using namespace std;
using namespace cellar;

class PieBoundary : public AbstractBoundary
{
public:
    PieBoundary() :
        AbstractBoundary("Pie Boundary"),
        vertex(-1, glm::dvec3(0, 0, 0))
    {

    }

    virtual ~PieBoundary()
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


PieDeserializer::PieDeserializer()
{

}

PieDeserializer::~PieDeserializer()
{

}

void logError(const std::string& msg)
{
    getLog().postMessage(new Message('E', false, msg, "PieDeserializer"));
}

void logInfo(const std::string& msg)
{
    getLog().postMessage(new Message('I', false, msg, "PieDeserializer"));
}

bool PieDeserializer::deserialize(
        const std::string& fileName,
        Mesh& mesh,
        std::vector<MeshMetric>& metrics) const
{
    PI_FICHIER_IMPORT pifich(fileName, "", 0);

    int rc = pifich.analyser_syntaxe();
    if(rc != 0)
    {
        logError("Syntax error (erno=" + to_string(rc) + ")");
        return false;
    }


    shared_ptr<PieBoundary> pieBoundary(
        make_shared<PieBoundary>());
    mesh.setBoundary(pieBoundary);


    logInfo("Mesh count : " + to_string(pifich.get_nbmaillages()));
    const PM_MAILLAGE* maillage = pifich.get_maillage(0);


    int nodeCount = maillage->get_nbsommets();
    mesh.verts.reserve(nodeCount);    
    mesh.topos.resize(nodeCount);

    glm::dvec3 minBox(1.0/0.0);
    glm::dvec3 maxBox(-1.0/0.0);
    for(int n=0; n < nodeCount; ++n)
    {
        int code = 0;
        double coord[3] = {0, 0, 0};
        maillage->get_sommet(n, coord, code);

        mesh.verts.push_back(MeshVert(
            glm::dvec3(coord[0], coord[1], coord[2])));

        minBox = glm::min(minBox, mesh.verts.back().p);
        maxBox = glm::max(maxBox, mesh.verts.back().p);
    }

    getLog().postMessage(new Message('I', false, "Original bounding box = (" +
        to_string(minBox.x) + ", " + to_string(minBox.y) + ", " + to_string(minBox.z) + ") to (" +
        to_string(maxBox.x) + ", " + to_string(maxBox.y) + ", " + to_string(maxBox.z) + ")",
        "Mesh"));

    glm::dvec3 ext = maxBox - minBox;
    double scale = 2.0 / glm::max(glm::max(ext.x, ext.y), ext.z);
    glm::dvec3 center = (minBox + maxBox) / 2.0;

    for(auto& v : mesh.verts)
    {
        v.p = (v.p - center) * scale;
    }

    int zoneCount = maillage->get_nbzones();
    for(int z=0; z < zoneCount; ++z)
    {
        const PM_ZONE* zone = maillage->get_zone(z);
        int elemCount = zone->get_nbelements();
        int elemVertCount = zone->get_nbnoeudselem();
        vector<int> connect( elemVertCount );

        const PT_INTERPOLANT_AVEC_GEOMETRIE& interp = zone->get_interpolant();
        if(zone->est_structuree_non_indexee())
        {
            logError("Non indexed structured zone not supported (ignored)");
            continue;
        }
        else if (zone->est_structuree_et_indexee())
        {
            logInfo("Indexed structured zone : " + zone->get_nomDeLIdentificateur());

            assert(elemVertCount == 8);

            const int nbNoeuds = zone->get_champ_coordonnees()
                ->get_champ_donnees()->get_champ_index()->get_nbtotal_groupes();

            vector<int> indirection;
            indirection.reserve( nbNoeuds + 1 );
            indirection.push_back( -1 );

            for ( int n = 0; n < nbNoeuds; ++n )
            {
                double val;
                zone->get_champ_coordonnees()->get_champ_donnees()
                    ->get_champ_index()->v_get_groupe( n, &val );
                indirection.push_back( static_cast<int>( val ) - 1 );
            }

            for ( int e = 0; e < elemCount; ++e )
            {
                // la connectivite de l'element i
                zone->get_cncelement( e, &connect[0] );

                for ( int j = 0; j < elemVertCount; ++j )
                    connect[j] = indirection.at( connect[j] );

                mesh.hexs.push_back(MeshHex(
                    connect[0], connect[1], connect[2], connect[3],
                    connect[4], connect[5], connect[6], connect[7]));
            }
        }
        else if(!zone->est_structuree())
        {
            const string idInterp = interp.get_nomDeLIdentificateur();

            logInfo("Unstructured zone : " + zone->get_nomDeLIdentificateur()
                    + " " + idInterp);

            for ( int e = 0; e < elemCount; e++ )
            {
               // la connectivite de l'element i
               zone->get_cncelement( e, &connect[0] );
               for(int& c : connect) --c;

               if ( idInterp == "LagrTrian03" )
               {
                    mesh.topos[connect[0]] = MeshTopo(
                        pieBoundary->fixedConstrait());
                    mesh.topos[connect[1]] = MeshTopo(
                        pieBoundary->fixedConstrait());
                    mesh.topos[connect[2]] = MeshTopo(
                        pieBoundary->fixedConstrait());
               }
               else if ( idInterp == "LagrQuadr04" )
               {
                    mesh.topos[connect[0]] = MeshTopo(
                        pieBoundary->fixedConstrait());
                    mesh.topos[connect[1]] = MeshTopo(
                        pieBoundary->fixedConstrait());
                    mesh.topos[connect[2]] = MeshTopo(
                        pieBoundary->fixedConstrait());
                    mesh.topos[connect[4]] = MeshTopo(
                        pieBoundary->fixedConstrait());
               }
               else if ( idInterp == "LagrTetra04" )
               {
                   mesh.tets.push_back(MeshTet(
                        connect[0], connect[1],
                        connect[2], connect[3]));
               }
               else if ( idInterp == "LagrPrism06" )
               {
                   mesh.pris.push_back(MeshPri(
                        connect[0], connect[1], connect[2],
                        connect[3], connect[4], connect[5]));
               }
               else if ( idInterp == "LagrHexae08" )
               {
                   mesh.hexs.push_back(MeshHex(
                        connect[0], connect[1], connect[2], connect[3],
                        connect[4], connect[5], connect[6], connect[7]));
               }
               else
               {
                   logError("Unsupported element type (ignored)");
               }
            }
        }
    }

    int vertCount = mesh.verts.size();
    metrics.resize(vertCount);

    int nbsol = pifich.get_nbsolutions();

    double metricScale = (1.0e6) / (scale*scale);

    if(nbsol > 0)
    {
        PS_SOLUTION* sol = pifich.get_solution(0);

        int nbVar = sol->get_nbvariables();
        for(int i=0; i < nbVar; ++i)
        {
            PS_VARIABLE* var = sol->get_variable(i);
            int nbComp = var->get_nbcomposantes();

            logInfo(var->get_nomDeLIdentificateur() + " : " +
                    to_string(nbComp));

            if(nbComp == 6)
            {
                for(int v=0; v < vertCount; ++v)
                {
                    double h[6];
                    var->get_champ_valeurs()->get_champ_donnees()
                            ->v_get_groupe(v, h);

                    metrics[v] = MeshMetric(
                        h[0], h[1], h[2],
                        h[1], h[3], h[4],
                        h[2], h[4], h[5]) * metricScale;
                }

            }
        }

    }

    return true;
}
