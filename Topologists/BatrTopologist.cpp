#include "BatrTopologist.h"

#include <CellarWorkbench/Misc/Log.h>

#include "TriangularBoundary.h"
#include "DataStructures/Mesh.h"
#include "DataStructures/MeshCrew.h"
#include "Measurers/AbstractMeasurer.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace cellar;

template<typename T, typename N>
inline std::vector<T> concat(const std::vector<T>& a, const std::vector<N>& b)
{
    std::vector<T> c(a.begin(), a.end());
    c.insert(c.end(), b.begin(), b.end());
    return std::move(c);
}

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
        "Performing BATR topology modifications...",
        "BatrTopologist"));

    edgeSplitMerge(mesh, crew);
    faceSwapping(mesh, crew);
    edgeSwapping(mesh, crew);
}

void BatrTopologist::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{

}

void BatrTopologist::edgeSplitMerge(
        Mesh& mesh,
        const MeshCrew& crew) const
{
    TriangularBoundary bounds(mesh.tets);

    std::vector<MeshVert>& verts = mesh.verts;
    std::vector<MeshTopo>& topos = mesh.topos;

    std::vector<bool> aliveVerts(verts.size(), true);
    std::vector<bool> vertsToTry(verts.size(), true);
    std::vector<bool> aliveTets(mesh.tets.size(), true);

    size_t passCount = 0;
    size_t vertMergeCount = 0;
    size_t edgeSplitCount = 0;
    bool stillVertsToTry = true;
    while(stillVertsToTry)
    {
        ++passCount;
        stillVertsToTry = false;


        for(size_t vId=0; vId < verts.size(); ++vId)
        {
            if(!aliveVerts[vId])
            {
                continue;
            }

            vertsToTry[vId] = false;

            double minDist = INFINITY;
            size_t minId = 0;

            double maxDist = -INFINITY;
            size_t maxId = 0;

            for(size_t nAr = 0; nAr < topos[vId].neighborVerts.size(); ++nAr)
            {
                MeshNeigVert nId = topos[vId].neighborVerts[nAr];

                //if(nId < vId)
                //{
                //    continue;
                //}

                MeshVert vert = verts[vId];
                MeshTopo& vTopo = topos[vId];

                MeshVert neig = verts[nId];
                MeshTopo& nTopo = topos[nId];

                if((vTopo.isFixed && (nTopo.isFixed || nTopo.isBoundary)) ||
                   (nTopo.isFixed && (vTopo.isFixed || vTopo.isBoundary)) ||
                   (vTopo.isBoundary && nTopo.isBoundary &&
                    vTopo.snapToBoundary != nTopo.snapToBoundary))
                {
                    continue;
                }


                double dist = crew.measurer().riemannianDistance(
                            crew.sampler(), vert.p, neig.p, vert.c);

                if(dist < minEdgeLength() && dist < minDist)
                {
                    minDist = dist;
                    minId = nId;
                }

                if(dist > maxEdgeLength() && dist > maxDist)
                {
                    maxDist = dist;
                    maxId = nId;
                }
            }

            uint nId;
            double dist;
            if(minDist != INFINITY)
            {
                dist = minDist;
                nId = minId;
            }
            else if(maxDist != -INFINITY)
            {
                dist = maxDist;
                nId = maxId;
            }
            else
            {
                continue;
            }


            MeshVert vert = verts[vId];
            MeshTopo& vTopo = topos[vId];

            MeshVert neig = verts[nId];
            MeshTopo& nTopo = topos[nId];


            // Merging vertices
            std::vector<MeshNeigElem>& vElems = vTopo.neighborElems;
            std::vector<MeshNeigElem>& nElems = nTopo.neighborElems;
            std::vector<MeshNeigElem> intersectionElems;

            // Element intersection
            for(const MeshNeigElem& vElem : vElems)
            {
                for(const MeshNeigElem& nElem : nElems)
                {
                    if(vElem.id == nElem.id)
                    {
                        intersectionElems.push_back(vElem);
                        break;
                    }
                }
            }

            // Elements from v not in intersection
            std::vector<MeshNeigElem> vExclusiveElems;
            for(const MeshNeigElem& vElem : vElems)
            {
                bool isIntersection = false;
                for(const MeshNeigElem& iElem : intersectionElems)
                {
                    if(vElem.id == iElem.id)
                    {
                        isIntersection = true;
                        break;
                    }
                }

                if(!isIntersection)
                    vExclusiveElems.push_back(vElem);
            }

            // Elements from n not in intersection
            std::vector<MeshNeigElem> nExclusiveElems;
            for(const MeshNeigElem& nElem : nElems)
            {
                bool isIntersection = false;
                for(const MeshNeigElem& iElem : intersectionElems)
                {
                    if(nElem.id == iElem.id)
                    {
                        isIntersection = true;
                        break;
                    }
                }

                if(!isIntersection)
                    nExclusiveElems.push_back(nElem);
            }


            std::vector<MeshNeigVert>& vVerts = vTopo.neighborVerts;
            std::vector<MeshNeigVert>& nVerts = nTopo.neighborVerts;

            // Find intersection and n's exclusive vertices
            std::vector<uint> nExclusiveVerts;
            std::vector<uint> intersectionVerts;
            for(const MeshNeigVert& nVert : nVerts)
            {
                if(nVert != vId)
                {
                    bool isExclusive = true;
                    for(const MeshNeigVert& vVert : vVerts)
                        if(nVert.v == vVert.v) {isExclusive = false; break;}

                    if(isExclusive)
                        nExclusiveVerts.push_back(nVert);
                    else
                        intersectionVerts.push_back(nVert);
                }
            }


            if(dist < minEdgeLength())
            {
                glm::dvec3 middle = (vert.p + neig.p) /2.0;
                if(vTopo.isBoundary)
                    middle = (*vTopo.snapToBoundary)(middle);
                else if(nTopo.isBoundary)
                    middle = (*nTopo.snapToBoundary)(middle);
                else if(vTopo.isFixed)
                    middle = vert.p;
                else if(nTopo.isFixed)
                    middle = neig.p;

                verts[vId].p = verts[nId].p = middle;


                std::vector<MeshNeigElem> allExclusiveElems(
                        vExclusiveElems.begin(),
                        vExclusiveElems.end());
                allExclusiveElems.insert(
                        allExclusiveElems.end(),
                        nExclusiveElems.begin(),
                        nExclusiveElems.end());

                bool isConformal = true;
                for(const MeshNeigElem& dElem : allExclusiveElems)
                {
                    if(crew.measurer().tetEuclideanVolume(
                        mesh, bounds.tet(dElem.id)) <= 0.0)
                    {
                        isConformal = false;
                        break;
                    }
                }

                if(!isConformal)
                {
                    // Rollback modifications
                    verts[vId].p = vert;
                    verts[nId].p = neig;
                    continue;
                }


                // Mark geometry as deleted
                aliveVerts[nId] = false;
                for(const MeshNeigElem& cElem : intersectionElems)
                {
                    aliveTets[cElem.id] = false;
                    bounds.removeTet(cElem.id);
                }

                // Find v's exclusive vertices
                std::vector<uint> vExclusiveVerts;
                for(const MeshNeigVert& vVert : vVerts)
                {
                    if(vVert != nId)
                    {
                        bool isExclusive = true;
                        for(const MeshNeigVert& iVert : intersectionVerts)
                            if(vVert.v == iVert.v) {isExclusive = false; break;}

                        if(isExclusive)
                            vExclusiveVerts.push_back(vVert);
                    }
                }

                // Replace n for v in n's exclusive elements
                for(const MeshNeigElem& ndElem : nExclusiveElems)
                    bounds.removeTet(ndElem.id);

                for(const MeshNeigElem& ndElem : nExclusiveElems)
                {
                    MeshTet tet = bounds.tet(ndElem.id);
                    if(tet.v[0] == nId)      tet.v[0] = vId;
                    else if(tet.v[1] == nId) tet.v[1] = vId;
                    else if(tet.v[2] == nId) tet.v[2] = vId;
                    else if(tet.v[3] == nId) tet.v[3] = vId;
                    bounds.insertTet(tet, ndElem.id);
                }

                // Replace n for v in n's exclusive vertices
                for(uint nVert : nExclusiveVerts)
                {
                    for(MeshNeigVert& neVert : topos[nVert].neighborVerts)
                        if(neVert.v == nId) {neVert.v = vId; break;}
                }


                // Rebuild v vert neighborhood
                size_t vCopyVerts = 0;
                for(size_t i=0; i < vVerts.size(); ++i)
                {
                    if(vVerts[i] != nId)
                    {
                        vVerts[vCopyVerts] = vVerts[i];
                        ++vCopyVerts;
                    }
                }
                vVerts.resize(vCopyVerts);
                vVerts = concat(vVerts, nExclusiveVerts);
                vVerts.shrink_to_fit();

                // Rebuild v elem neighborhood
                vElems = allExclusiveElems;
                vElems.shrink_to_fit();


                // Update intersection vertices' topo
                std::set<uint> interSet;
                for(const MeshNeigElem& iElem : intersectionElems)
                    interSet.insert(iElem.id);

                for(uint iId : intersectionVerts)
                {
                    // Remove intersection elems from intersection verts
                    size_t elemCopyId = 0;
                    std::vector<MeshNeigElem>& elemsCopy = topos[iId].neighborElems;
                    for(size_t i=0; i < elemsCopy.size(); ++i)
                    {
                        if(interSet.find(elemsCopy[i].id) == interSet.end())
                        {
                            elemsCopy[elemCopyId] = elemsCopy[i];
                            ++elemCopyId;
                        }
                    }
                    elemsCopy.resize(elemCopyId);
                    elemsCopy.shrink_to_fit();

                    // Remove n from intersection verts
                    size_t vertCopyId = 0;
                    std::vector<MeshNeigVert>& vertCopy = topos[iId].neighborVerts;
                    for(size_t i=0; i < vertCopy.size(); ++i)
                    {
                        if(vertCopy[i] != nId)
                        {
                            vertCopy[vertCopyId] = vertCopy[i];
                            ++vertCopyId;
                        }
                    }
                    vertCopy.resize(vertCopyId);
                    vertCopy.shrink_to_fit();
                }


                // Update boundary status
                if(nTopo.isFixed)
                {
                    vTopo.isFixed = true;
                }
                else if(!vTopo.isBoundary && nTopo.isBoundary)
                {
                    vTopo.isBoundary = true;
                    vTopo.snapToBoundary = nTopo.snapToBoundary;
                }


                // Notify to check neighbor verts
                ++vertMergeCount;
                stillVertsToTry = true;
                vertsToTry[vId] = true;
                for(const MeshNeigVert vVert : vTopo.neighborVerts)
                    vertsToTry[vVert.v] = true;
            }
            else if(dist > maxEdgeLength())
            {
                // Splitting the edge
                uint wId = verts.size();
                glm::dvec3 middle = (vert.p + neig.p) /2.0;
                verts.push_back(MeshVert(middle, vert.c));

                MeshTopo wTopo;
                if(vTopo.isBoundary && nTopo.isBoundary)
                {
                    if(bounds.isBoundary(vId, nId))
                    {
                        wTopo.isBoundary = true;
                        wTopo.snapToBoundary = vTopo.snapToBoundary;
                        verts[wId].p = (*vTopo.snapToBoundary)(middle);
                    }
                }

                // Replace n for w in v's neighbor verts
                for(MeshNeigVert& vVert : vTopo.neighborVerts)
                    if(vVert == nId) { vVert.v = wId; break; }

                // Replace v for w in n's neighbor verts
                for(MeshNeigVert& nVert : nTopo.neighborVerts)
                    if(nVert == vId) { nVert.v = wId; break; }


                // Remove intersection tets from bounds
                for(const MeshNeigElem& iElem : intersectionElems)
                    bounds.removeTet(iElem.id);


                // Build new elements
                nElems = nExclusiveElems;
                std::vector<MeshNeigElem>& wElems = wTopo.neighborElems;
                for(const MeshNeigElem& iElem : intersectionElems)
                {
                    int o = -1;
                    uint others[] = {0, 0};
                    MeshTet tet = bounds.tet(iElem.id);
                    if(tet.v[0] != vId)
                        if(tet.v[0] == nId) tet.v[0] = wId;
                        else others[++o] = tet.v[0];
                    if(tet.v[1] != vId)
                        if(tet.v[1] == nId) tet.v[1] = wId;
                        else others[++o] = tet.v[1];
                    if(tet.v[2] != vId)
                        if(tet.v[2] == nId) tet.v[2] = wId;
                        else others[++o] = tet.v[2];
                    if(tet.v[3] != vId)
                        if(tet.v[3] == nId) tet.v[3] = wId;
                        else others[++o] = tet.v[3];


                    MeshNeigElem newNeigElem(MeshTet::ELEMENT_TYPE, bounds.tetCount());
                    nElems.push_back(newNeigElem);
                    wElems.push_back(newNeigElem);
                    wElems.push_back(iElem);

                    topos[others[0]].neighborElems.push_back(newNeigElem);
                    topos[others[1]].neighborElems.push_back(newNeigElem);


                    MeshTet newTetN(nId, wId, others[0], others[1]);
                    if(AbstractMeasurer::tetEuclideanVolume(mesh, newTetN) < 0.0)
                        std::swap(newTetN.v[0], newTetN.v[1]);
                    if(AbstractMeasurer::tetEuclideanVolume(mesh, tet) < 0.0)
                        std::swap(tet.v[0], tet.v[1]);

                    bounds.insertTet(tet, iElem.id);
                    bounds.insertTet(newTetN);
                    aliveTets.push_back(true);
                }

                // Add w to intersection nodes' neighbors
                for(uint iVert : intersectionVerts)
                    topos[iVert].neighborVerts.push_back(wId);

                // Build w's vert neighbors list
                std::vector<MeshNeigVert>& wVerts = wTopo.neighborVerts;
                wVerts = concat(wVerts, intersectionVerts);
                wVerts.push_back(vId);
                wVerts.push_back(nId);


                // Notify to check neighbor verts
                ++edgeSplitCount;
                stillVertsToTry = true;
                for(const MeshNeigVert vVert : vTopo.neighborVerts)
                    vertsToTry[vVert.v] = true;
                for(const MeshNeigVert nVert : nTopo.neighborVerts)
                    vertsToTry[nVert.v] = true;

                // Push back only at the end cause we have
                // lots of active references to topo all around
                // (prevent vector from reallocating its buffer)
                aliveVerts.push_back(true);
                vertsToTry.push_back(true);
                topos.push_back(wTopo);
            }
        }
    }


    // Remove deleted tets
    size_t copyTetId = 0;
    size_t tetCount = bounds.tetCount();
    std::vector<MeshTet>& tets = mesh.tets;
    tets.resize(tetCount);
    for(size_t tId=0; tId < tetCount; ++tId)
    {
        if(aliveTets[tId])
        {
            const MeshLocalTet& tet = bounds.tet(tId);
            for(uint i=0; i < 4; ++i)
            {
                MeshTopo& topo = topos[tet.v[i]];
                for(MeshNeigElem& ne : topo.neighborElems)
                    if(ne.id == tId) {ne.id = copyTetId; break;}
            }

            tets[copyTetId] = tet;
            ++copyTetId;
        }
    }
    tets.resize(copyTetId);


    // Remove deleted verts
    size_t copyVertId = 0;
    for(size_t vId=0; vId < verts.size(); ++vId)
    {
        if(aliveVerts[vId])
        {
            if(copyVertId != vId)
            {
                // Update neighbor verts lists
                for(MeshNeigVert& nv : topos[vId].neighborVerts)
                    for(MeshNeigVert& nnv : topos[nv.v].neighborVerts)
                        if(nnv.v == vId) {nnv.v = copyVertId; break;}

                // Update neighbor elems lists
                for(MeshNeigElem& ne : topos[vId].neighborElems)
                {
                    MeshTet& tet = tets[ne.id];
                    if(tet.v[0] == vId) tet.v[0] = copyVertId;
                    else if(tet.v[1] == vId) tet.v[1] = copyVertId;
                    else if(tet.v[2] == vId) tet.v[2] = copyVertId;
                    else if(tet.v[3] == vId) tet.v[3] = copyVertId;
                }

                verts[copyVertId] = verts[vId];
                topos[copyVertId] = topos[vId];
            }
            ++copyVertId;
        }
    }
    verts.resize(copyVertId);
    topos.resize(copyVertId);


    getLog().postMessage(new Message('I', false,
        "Edge split/merge: " +
        std::to_string(passCount) + " passes \t(" +
        std::to_string(edgeSplitCount) + " split, " +
        std::to_string(vertMergeCount) + " merge)",
        "BatrTopologist"));
}

void BatrTopologist::faceSwapping(
        Mesh& mesh,
        const MeshCrew& crew) const
{
    std::vector<MeshTet>& tets = mesh.tets;
    std::vector<MeshTopo>& topos = mesh.topos;

    size_t passCount = 0;
    size_t faceSwapCount = 0;
    bool stillTetsToTry = true;

    std::vector<bool> tetsToTry(tets.size(), true);
    while(stillTetsToTry)
    {
        ++passCount;
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

                            // TODO wbussiere 2016-04-20 :
                            // Expand 'to try' marker to neighboor tets
                            stillTetsToTry = true;
                            tetsToTry.push_back(true);
                            tetsToTry[nt] = true;
                            tetsToTry[t] = true;
                            ++faceSwapCount;
                            break;
                        }
                    }
                }
            }
        }
    }

    getLog().postMessage(new Message('I', false,
        "Face swap:        " +
        std::to_string(passCount) + " passes \t(" +
        std::to_string(faceSwapCount) + " swap)",
        "BatrTopologist"));

    mesh.compileTopology(false);
}

void BatrTopologist::edgeSwapping(
        Mesh& mesh,
        const MeshCrew& crew) const
{

}
