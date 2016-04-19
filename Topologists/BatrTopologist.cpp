#include "BatrTopologist.h"

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace cellar;


BatrTopologist::BatrTopologist()
{

}

BatrTopologist::~BatrTopologist()
{

}

bool BatrTopologist::needTopologicalModifications(
            int vertRelocationPassCount,
            const Mesh& mesh) const
{
    if(mesh.tets.empty() || !(mesh.pris.empty() && mesh.hexs.empty()))
        return false;

    return isEnabled() &&
           (vertRelocationPassCount > 1) &&
           ((vertRelocationPassCount-1) % frequency() == 0);
}

void BatrTopologist::restructureMesh(
        Mesh& mesh,
        const MeshCrew& crew) const
{
    getLog().postMessage(new Message('I', false,
        "Performing new BATR topology modifications",
        "BatrTopologist"));

    edgeSplitting(mesh, crew);
    faceSwapping(mesh, crew);
    edgeSwapping(mesh, crew);
}

void BatrTopologist::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{

}


void BatrTopologist::edgeSplitting(
        Mesh& mesh,
        const MeshCrew& crew) const
{

}

void BatrTopologist::faceSwapping(
        Mesh& mesh,
        const MeshCrew& crew) const
{
    std::vector<MeshTet>& tets = mesh.tets;
    std::vector<MeshTopo>& topos = mesh.topos;

    uint faceSwap = 0;
    bool stillTetsToTry = true;
    std::vector<bool> tetsToTry(tets.size(), true);
    while(stillTetsToTry)
    {
        stillTetsToTry = false;

        for(size_t t=0; t < tets.size(); ++t)
        {
            if(!tetsToTry[t])
                continue;

            tetsToTry[t] = false;
            const MeshTet& tet = tets[t];
            for(uint f=0; f < MeshTet::TRI_COUNT; ++f)
            {
                const MeshTri& refTri = MeshTet::tris[f];
                MeshTri tri(tet.v[refTri[0]], tet.v[refTri[1]], tet.v[refTri[2]]);

                if(tri.v[0] < tri.v[1] && tri.v[1] < tri.v[2] ||
                   tri.v[1] < tri.v[2] && tri.v[2] < tri.v[0] ||
                   tri.v[2] < tri.v[0] && tri.v[0] < tri.v[1])
                {
                    uint nb = 0;
                    uint common[] = {0, 0};

                    std::vector<MeshNeigElem>& neigTets0 = topos[tri.v[0]].neighborElems;
                    std::vector<MeshNeigElem>& neigTets1 = topos[tri.v[1]].neighborElems;
                    std::vector<MeshNeigElem>& neigTets2 = topos[tri.v[2]].neighborElems;
                    for(const MeshNeigElem& n0 : neigTets0)
                    {
                        bool isIn = false;
                        for(const MeshNeigElem& n1 : neigTets1)
                        {
                            if(n0.id == n1.id)
                            {
                                isIn = true;
                                break;
                            }
                        }

                        if(isIn)
                        {
                            isIn = false;
                            for(const MeshNeigElem& n2 : neigTets2)
                            {
                                if(n0.id == n2.id)
                                {
                                    isIn = true;
                                    break;
                                }
                            }

                            if(isIn)
                            {
                                common[nb] = n0.id;
                                ++nb;
                            }
                        }
                    }

                    if(nb == 2)
                    {
                        uint tOp = tet.v[f];

                        uint nt  = ((t != common[0]) ? common[0] : common[1]);
                        const MeshTet& ntet = tets[nt];
                        uint nOp = ntet.v[0];

                        if(ntet.v[1] != tri.v[0] && ntet.v[1] != tri.v[1] && ntet.v[1] != tri.v[2])
                            nOp = ntet.v[1];
                        else if(ntet.v[2] != tri.v[0] && ntet.v[2] != tri.v[1] && ntet.v[2] != tri.v[2])
                            nOp = ntet.v[2];
                        else if(ntet.v[3] != tri.v[0] && ntet.v[3] != tri.v[1] && ntet.v[3] != tri.v[2])
                            nOp = ntet.v[3];


                        double minQual = crew.evaluator().tetQuality(
                            mesh, crew.sampler(), crew.measurer(), tet);
                        minQual = glm::min(minQual, crew.evaluator().tetQuality(
                            mesh, crew.sampler(), crew.measurer(), ntet));

                        MeshTet newTet0(tOp, tri[0], tri[1], nOp, tet.c[0]);
                        MeshTet newTet1(tOp, tri[1], tri[2], nOp, tet.c[0]);
                        MeshTet newTet2(tOp, tri[2], tri[0], nOp, tet.c[0]);

                        if(minQual < crew.evaluator().tetQuality(mesh,
                                crew.sampler(), crew.measurer(), newTet0) &&
                           minQual < crew.evaluator().tetQuality(mesh,
                                crew.sampler(), crew.measurer(), newTet1) &&
                           minQual < crew.evaluator().tetQuality(mesh,
                                crew.sampler(), crew.measurer(), newTet2))
                        {
                            tets[t] = newTet0;
                            tets[nt] = newTet1;
                            uint lt = tets.size();
                            tets.push_back(newTet2);

                            topos[tOp].neighborElems.push_back(MeshNeigElem(MeshTet::ELEMENT_TYPE, nt));
                            topos[tOp].neighborElems.push_back(MeshNeigElem(MeshTet::ELEMENT_TYPE, lt));

                            topos[nOp].neighborElems.push_back(MeshNeigElem(MeshTet::ELEMENT_TYPE, t));
                            topos[nOp].neighborElems.push_back(MeshNeigElem(MeshTet::ELEMENT_TYPE, lt));

                            for(MeshNeigElem& v : neigTets0)
                                if(v.id == nt) {v.id = lt; break;}

                            for(MeshNeigElem& v : neigTets2)
                                if(v.id == t) {v.id = lt; break;}

                            stillTetsToTry = true;
                            tetsToTry.push_back(true);
                            tetsToTry[nt] = true;
                            tetsToTry[t] = true;
                            ++faceSwap;
                            break;
                        }
                    }
                }
            }
        }
    }

    mesh.compileTopology(false);

    getLog().postMessage(new Message('I', false,
        "Face swaps : " + std::to_string(faceSwap),
        "BatrTopologist"));
}

void BatrTopologist::edgeSwapping(
        Mesh& mesh,
        const MeshCrew& crew) const
{

}
