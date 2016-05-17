#include "BatrTopologist.h"

#include <list>
#include <iostream>
#include <algorithm>

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/MeshCrew.h"
#include "Measurers/AbstractMeasurer.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace cellar;


inline MeshNeigElem toTet(uint eId)
{
    return MeshNeigElem(MeshTet::ELEMENT_TYPE, eId);
}

inline void appendElems(std::vector<MeshNeigElem>& elems, const std::vector<uint>& eIds)
{
    elems.reserve(elems.size() + eIds.size());
    for(uint eId : eIds) elems.push_back(toTet(eId));
}

inline void appendVerts(std::vector<MeshNeigVert>& verts, const std::vector<uint>& vIds)
{
    verts.reserve(verts.size() + vIds.size());
    for(uint vId : vIds) verts.push_back(vId);
}


BatrTopologist::BatrTopologist()
{
    _ringConfigDictionary = {
      {}, {}, {},
      // 3
      {RingConfig(1, {MeshTri(0, 1, 2)})},
      // 4
      {RingConfig(2, {MeshTri(0, 1, 2), MeshTri(0, 2, 3)})},
      // 5
      {RingConfig(5, {MeshTri(0, 1, 3), MeshTri(0, 3, 4), MeshTri(1, 2, 3)})},
      // 6
      {RingConfig(6, {MeshTri(0, 1, 2), MeshTri(0, 2, 3), MeshTri(0, 3, 4), MeshTri(0, 4, 5)}),
       RingConfig(3, {MeshTri(0, 1, 2), MeshTri(0, 2, 3), MeshTri(0, 3, 5), MeshTri(3, 4, 5)}),
       RingConfig(3, {MeshTri(0, 1, 3), MeshTri(1, 2, 3), MeshTri(0, 3, 4), MeshTri(0, 4, 5)}),
       RingConfig(2, {MeshTri(0, 1, 2), MeshTri(0, 2, 4), MeshTri(2, 3, 4), MeshTri(0, 4, 5)})},
      // 7
      {RingConfig(7, {MeshTri(0, 1, 2), MeshTri(0, 2, 3), MeshTri(0, 3, 4), MeshTri(0, 4, 5), MeshTri(0, 5, 6)}),
       RingConfig(7, {MeshTri(0, 1, 2), MeshTri(0, 2, 3), MeshTri(0, 3, 6), MeshTri(3, 4, 6), MeshTri(4, 5, 6)}),
       RingConfig(7, {MeshTri(0, 1, 2), MeshTri(0, 2, 3), MeshTri(0, 3, 4), MeshTri(0, 4, 6), MeshTri(4, 5, 6)}),
       RingConfig(7, {MeshTri(0, 1, 3), MeshTri(1, 2, 3), MeshTri(0, 3, 4), MeshTri(0, 4, 5), MeshTri(0, 5, 6)}),
       RingConfig(7, {MeshTri(0, 1, 2), MeshTri(0, 2, 4), MeshTri(2, 3, 4), MeshTri(0, 4, 5), MeshTri(0, 5, 6)}),
       RingConfig(7, {MeshTri(0, 1, 2), MeshTri(0, 2, 3), MeshTri(0, 3, 5), MeshTri(3, 4, 5), MeshTri(0, 5, 6)})}
    };
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

    while(true)
    {
        if(!edgeSplitMerge(mesh, crew)) break;
        if(!faceSwapping(mesh, crew))   break;
        if(!edgeSwapping(mesh, crew))   break;
    }

    mesh.compileTopology(false);
}

void BatrTopologist::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{

}

bool BatrTopologist::edgeSplitMerge(
        Mesh& mesh,
        const MeshCrew& crew) const
{
    std::vector<MeshTet>& tets = mesh.tets;
    std::vector<MeshVert>& verts = mesh.verts;
    std::vector<MeshTopo>& topos = mesh.topos;

    bool stillVertsToTry = true;
    std::vector<bool> aliveVerts(verts.size(), true);
    std::vector<bool> vertsToTry(verts.size(), true);
    std::vector<bool> aliveTets(mesh.tets.size(), true);

    size_t passCount = 0;
    size_t maxPassCount = 20;
    size_t vertMergeCount = 0;
    size_t edgeSplitCount = 0;
    for(;passCount < maxPassCount && stillVertsToTry; ++passCount)
    {
        stillVertsToTry = false;

        size_t passVertMergeCount = 0;
        size_t passEdgeSplitCount = 0;


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

                if(nId < vId)
                {
                    continue;
                }

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
            else
            if(maxDist != -INFINITY)
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


            // Find verts and elems ring
            std::vector<uint> ringVerts;
            std::vector<uint> ringElems;
            findRing(tets, mesh.topos, vId, nId, ringVerts, ringElems);

            // Elements from n not in ring
            std::vector<uint> nExElems;
            findExclusiveElems(mesh, nId, ringElems, nExElems);


            if(dist < minEdgeLength())
            {
                // Elements from v not in ring
                std::vector<uint> vExElems;
                findExclusiveElems(mesh, vId, ringElems, vExElems);

                std::vector<uint> allExElems(vExElems.begin(), vExElems.end());
                allExElems.insert(allExElems.end(), nExElems.begin(), nExElems.end());


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

                bool isConformal = true;
                for(uint dElem : allExElems)
                {
                    if(crew.measurer().tetEuclideanVolume(
                        mesh, tets[dElem]) <= 0.0)
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

                std::vector<uint> vExVerts;
                findExclusiveVerts(mesh, vId, nId, ringVerts, vExVerts);

                std::vector<uint> nExVerts;
                findExclusiveVerts(mesh, nId, vId, ringVerts, nExVerts);


                // Mark geometry as deleted
                aliveVerts[nId] = false;
                for(uint rElem : ringElems)
                    aliveTets[rElem] = false;

                // Replace n for v in n's exclusive elements
                for(uint ndElem : nExElems)
                {
                    MeshTet& tet = tets[ndElem];
                    if(tet.v[0] == nId)      tet.v[0] = vId;
                    else if(tet.v[1] == nId) tet.v[1] = vId;
                    else if(tet.v[2] == nId) tet.v[2] = vId;
                    else if(tet.v[3] == nId) tet.v[3] = vId;
                }

                // Replace n for v in n's exclusive vertices
                for(uint nVert : nExVerts)
                {
                    for(MeshNeigVert& neVert : topos[nVert].neighborVerts)
                        if(neVert.v == nId) {neVert.v = vId; break;}
                }

                // Remove n from ring verts
                // Remove ring elems from ring verts
                for(uint rVert : ringVerts)
                {
                    popOut(topos[rVert].neighborVerts, MeshNeigVert(nId));
                    popOut(topos[rVert].neighborElems, ringElems);
                }


                // Rebuild v vert neighborhood
                std::vector<MeshNeigVert>& vVerts = vTopo.neighborVerts;
                vVerts.clear();
                appendVerts(vVerts, vExVerts);
                appendVerts(vVerts, ringVerts);
                appendVerts(vVerts, nExVerts);
                vVerts.shrink_to_fit();

                // Rebuild v elem neighborhood
                std::vector<MeshNeigElem>& vElems = vTopo.neighborElems;
                vElems.clear();
                appendElems(vElems, vExElems);
                appendElems(vElems, nExElems);
                vElems.shrink_to_fit();

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
                ++passVertMergeCount;
                stillVertsToTry = true;
                vertsToTry[vId] = true;
                vertsToTry[nId] = false;
                for(const MeshNeigVert vVert : vTopo.neighborVerts)
                    vertsToTry[vVert.v] = true;
            }
            else if(dist > maxEdgeLength())
            {
                // Splitting the edge
                uint wId = verts.size();
                glm::dvec3 middle = (vert.p + neig.p) /2.0;

                MeshTopo wTopo;
                if(ringElems.size() < ringVerts.size())
                {
                    wTopo.isBoundary = true;
                    wTopo.snapToBoundary = vTopo.snapToBoundary;
                    middle = (*vTopo.snapToBoundary)(middle);
                }

                verts.push_back(MeshVert(middle, vert.c));

                // Replace n for w in v's neighbor verts
                for(MeshNeigVert& vVert : vTopo.neighborVerts)
                    if(vVert == nId) { vVert.v = wId; break; }

                // Replace v for w in n's neighbor verts
                for(MeshNeigVert& nVert : nTopo.neighborVerts)
                    if(nVert == vId) { nVert.v = wId; break; }


                // Build new elements
                std::vector<MeshNeigElem>& wElems = wTopo.neighborElems;
                std::vector<MeshNeigElem>& nElems = nTopo.neighborElems;

                nElems.clear();
                appendElems(nElems, nExElems);
                for(uint rElem : ringElems)
                {
                    int o = -1;
                    uint others[] = {0, 0};
                    MeshTet tet = tets[rElem];
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


                    wElems.push_back(toTet(rElem));
                    MeshNeigElem newElem = toTet(tets.size());
                    nElems.push_back(newElem);
                    wElems.push_back(newElem);

                    topos[others[0]].neighborElems.push_back(newElem);
                    topos[others[1]].neighborElems.push_back(newElem);


                    MeshTet newTetN(nId, wId, others[0], others[1]);
                    if(AbstractMeasurer::tetEuclideanVolume(mesh, newTetN) < 0.0)
                        std::swap(newTetN.v[0], newTetN.v[1]);
                    if(AbstractMeasurer::tetEuclideanVolume(mesh, tet) < 0.0)
                        std::swap(tet.v[0], tet.v[1]);

                    tets[rElem] = tet;
                    tets.push_back(newTetN);
                    aliveTets.push_back(true);
                }

                // Connect w to ring verts
                std::vector<MeshNeigVert>& wVerts = wTopo.neighborVerts;
                for(uint rVert : ringVerts)
                {
                    topos[rVert].neighborVerts.push_back(wId);
                    wVerts.push_back(rVert);
                }
                wVerts.push_back(vId);
                wVerts.push_back(nId);


                // Notify to check neighbor verts
                ++passEdgeSplitCount;
                stillVertsToTry = true;
                aliveVerts.push_back(true);
                vertsToTry.push_back(true);
                for(const MeshNeigVert& wVert : wVerts)
                    vertsToTry[wVert.v] = true;

                // Push back only at the end cause we have
                // lots of active references to topo all around
                // (prevent vector from reallocating its buffer)
                topos.push_back(wTopo);
            }
        }

        vertMergeCount += passVertMergeCount;
        edgeSplitCount += passEdgeSplitCount;
    }


    // Remove deleted tets
    size_t copyTetId = 0;
    size_t tetCount = tets.size();
    for(size_t tId=0; tId < tetCount; ++tId)
    {
        if(aliveTets[tId])
        {
            if(copyTetId != tId)
            {
                const MeshTet& tet = tets[tId];
                for(uint i=0; i < 4; ++i)
                {
                    MeshTopo& topo = topos[tet.v[i]];
                    for(MeshNeigElem& ne : topo.neighborElems)
                        if(ne.id == tId) {ne.id = copyTetId; break;}
                }

                tets[copyTetId] = tet;
            }
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

    return edgeSplitCount | vertMergeCount;
}

bool BatrTopologist::faceSwapping(
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

                // Process only counter-clockwise winded triangles
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

    return faceSwapCount;
}

bool BatrTopologist::edgeSwapping(
        Mesh& mesh,
        const MeshCrew& crew) const
{
    std::vector<MeshTet>& tets = mesh.tets;
    std::vector<MeshVert>& verts = mesh.verts;
    std::vector<MeshTopo>& topos = mesh.topos;

    bool stillVertsToTry = true;
    std::vector<bool> vertsToTry(verts.size(), true);
    std::vector<bool> aliveTets(tets.size(), true);
    std::vector<uint> deadTets;

    std::map<int, int> ringSizeCounters;
    std::map<int, int> edgeSwapCounters;
    size_t totalEdgeSwapCount = 0;
    size_t passCount = 0;

    while(stillVertsToTry)
    {
        ++passCount;
        stillVertsToTry = false;

        for(size_t vId=0; vId < verts.size(); ++vId)
        {
            if(!vertsToTry[vId])
                continue;

            vertsToTry[vId] = false;
            MeshTopo& vTopo = topos[vId];
            std::vector<MeshNeigVert>& vVerts = vTopo.neighborVerts;
            std::vector<MeshNeigElem>& vElems = vTopo.neighborElems;

            for(size_t nAr = 0; nAr < vVerts.size(); ++nAr)
            {
                uint nId = vVerts[nAr];

                if(nId < vId)
                {
                    continue;
                }

                std::vector<uint> ringVerts;
                std::vector<uint> ringElems;
                findRing(mesh.tets, mesh.topos, vId, nId, ringVerts, ringElems);

                // Sort ring verts
                std::vector<uint> elemPool = ringElems;

                int notFoundCount = 0;
                auto rIt = ringVerts.begin();
                auto nIt = ++ringVerts.begin();
                while(nIt != ringVerts.end())
                {
                    bool found = false;
                    for(size_t i=0; i < elemPool.size() && !found; ++i)
                    {
                        const MeshTet& tet = tets[elemPool[i]];
                        for(size_t j=0; j < 4 && !found; ++j)
                        {
                            if(tet.v[j] == *rIt)
                            {
                                for(size_t k=0; k < 4 && !found; ++k)
                                {
                                    if(k!=j && tet.v[k]!=vId && tet.v[k]!=nId)
                                    {
                                        auto wIt = nIt;
                                        while(wIt != ringVerts.end() && !found)
                                        {
                                            if(*wIt == tet.v[k])
                                            {
                                                std::swap(elemPool[i], elemPool.back());
                                                elemPool.pop_back();

                                                std::swap(*wIt, *nIt);
                                                found = true;
                                            }

                                            ++wIt;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // We've reached a bound
                    if(!found)
                    {
                        ++notFoundCount;

                        if(notFoundCount >= 2)
                        {
                            getLog().postMessage(new Message('W', false,
                                "Cannot close tet ring", "BatrTopologist"));
                            break;
                        }

                        auto cbIt = ringVerts.begin();
                        auto cfIt = rIt;

                        while(cbIt < cfIt)
                        {
                            std::swap(*cbIt, *cfIt);
                            ++cbIt; --cfIt;
                        }
                    }
                    else
                    {
                        ++rIt;
                        ++nIt;
                    }
                }

                if(nIt != ringVerts.end())
                    continue;


                // Verify that that this ring size can be handled
                size_t ringSize = ringVerts.size();
                ++ringSizeCounters[ringSize];
                if(ringSize < 3 || ringSize >= _ringConfigDictionary.size())
                {
                    ++edgeSwapCounters[0];
                    continue;
                }


                // Enforce counter clockwise order around segment V-N
                glm::dvec3 ringAir;
                glm::dvec3 ring0Pos = verts[ringVerts[0]].p;
                for(int i=2; i < ringSize; ++i)
                {
                    ringAir += glm::cross(
                        verts[ringVerts[i-1]].p - ring0Pos,
                        verts[ringVerts[i]].p   - ring0Pos);
                }

                glm::dvec3 vPos = verts[vId].p;
                glm::dvec3 nPos = verts[nId].p;
                glm::dvec3 vn = nPos - vPos;
                if(glm::dot(ringAir, vn) < 0.0)
                {
                    int i=0, j = ringVerts.size()-1;
                    while(i < j)
                    {
                        std::swap(ringVerts[i], ringVerts[j]);
                        ++i; --j;
                    }
                }


                // Compute current ring quality
                double minQual = INFINITY;
                for(uint eId : ringElems)
                {
                    minQual = glm::min(minQual, crew.evaluator().tetQuality(
                        mesh, crew.sampler(), crew.measurer(), tets[eId]));
                }


                // Check for outsider tets
                bool outsidersOk = true;
                std::vector<MeshTet> outsiderTets;
                for(int i=1; i <= ringSize; ++i)
                {
                    uint aId = ringVerts[i-1];
                    uint bId = ringVerts[i%ringSize];
                    glm::dvec3 normal = glm::cross(
                                verts[aId].p - vPos,
                                verts[bId].p - vPos);
                    double dotNormal = glm::dot(vn, normal);
                    if(dotNormal < 0.0)
                    {
                        MeshTet outsider(vId, bId, aId, nId);
                        if(minQual < crew.evaluator().tetQuality(mesh,
                            crew.sampler(), crew.measurer(), outsider))
                        {
                            outsiderTets.push_back(outsider);
                        }
                        else
                        {
                            // Removing segment V-N can't increase ring's quality
                            outsidersOk = false;
                            break;
                        }
                    }
                }

                if(!outsidersOk)
                    continue;


                // Compare each ring configuration's quality to actual quality
                int bestRingConfig = -1;
                int bestRingConfigRot = -1;
                const std::vector<RingConfig>& ringConfigs =
                        _ringConfigDictionary[ringVerts.size()];

                size_t ringConfigCount = ringConfigs.size();
                for(size_t conf = 0; conf < ringConfigCount; ++conf)
                {
                    const RingConfig& ringConfig = ringConfigs[conf];
                    for(uint rot = 0; rot < ringConfig.rotCount; ++rot)
                    {
                        double configRotMinQual = INFINITY;
                        for(const MeshTri& refTri : ringConfig.tris)
                        {
                            MeshTri tri(
                                ringVerts[(refTri.v[0] + rot) % ringSize],
                                ringVerts[(refTri.v[1] + rot) % ringSize],
                                ringVerts[(refTri.v[2] + rot) % ringSize]);

                            double tetVQual = crew.evaluator().tetQuality(
                                mesh, crew.sampler(), crew.measurer(),
                                MeshTet(tri.v[1], tri.v[0], tri.v[2], vId));

                            configRotMinQual = glm::min(configRotMinQual, tetVQual);
                            if(tetVQual < minQual) break;

                            double tetNQual = crew.evaluator().tetQuality(
                                mesh, crew.sampler(), crew.measurer(),
                                MeshTet(tri.v[0], tri.v[1], tri.v[2], nId));

                            configRotMinQual = glm::min(configRotMinQual, tetNQual);
                            if(tetNQual < minQual) break;
                        }

                        if(configRotMinQual > minQual)
                        {
                            bestRingConfig = conf;
                            bestRingConfigRot = rot;
                            minQual = configRotMinQual;
                        }
                    }
                }


                // Check if we found a better configuration
                if(bestRingConfig >= 0)
                {
                    int key = ringSize * 1000;
                    key += bestRingConfig * 100;
                    key += bestRingConfigRot * 10;
                    key += outsiderTets.size();
                    if(ringElems.size() < ringVerts.size())
                        key += 5;

                    ++edgeSwapCounters[key];
                }
                else
                {
                    ++edgeSwapCounters[-int(ringSize)];
                    continue;
                }


                // Finaly, update topology
                MeshTopo& nTopo = topos[nId];
                std::vector<MeshNeigVert>& nVerts = nTopo.neighborVerts;
                std::vector<MeshNeigElem>& nElems = nTopo.neighborElems;

                popOut(vElems, ringElems);
                popOut(nElems, ringElems);

                if(outsiderTets.empty())
                {
                    popOut(vVerts, nId);
                    popOut(nVerts, vId);
                }

                for(size_t i=0; i < ringVerts.size(); ++i)
                {
                    MeshTopo& rTopo = topos[ringVerts[i]];
                    popOut(rTopo.neighborElems, ringElems);
                }

                for(uint rElem : ringElems)
                {
                    aliveTets[rElem] = false;
                    deadTets.push_back(rElem);
                }

                std::vector<MeshTet> bestTets = outsiderTets;
                const RingConfig& besConfig = ringConfigs[bestRingConfig];
                for(const MeshTri& refTri : besConfig.tris)
                {
                    MeshTri tri(
                        ringVerts[(refTri.v[0] + bestRingConfigRot) % ringSize],
                        ringVerts[(refTri.v[1] + bestRingConfigRot) % ringSize],
                        ringVerts[(refTri.v[2] + bestRingConfigRot) % ringSize]);

                    MeshTet tetV(tri.v[1], tri.v[0], tri.v[2], vId);
                    bestTets.push_back(tetV);

                    MeshTet tetN(tri.v[0], tri.v[1], tri.v[2], nId);
                    bestTets.push_back(tetN);

                    make_union(topos[tri.v[0]].neighborVerts, tri.v[1]);
                    make_union(topos[tri.v[0]].neighborVerts, tri.v[2]);
                    make_union(topos[tri.v[1]].neighborVerts, tri.v[0]);
                    make_union(topos[tri.v[1]].neighborVerts, tri.v[2]);
                    make_union(topos[tri.v[2]].neighborVerts, tri.v[0]);
                    make_union(topos[tri.v[2]].neighborVerts, tri.v[1]);
                }

                for(const MeshTet& tet : bestTets)
                {
                    uint tetId;
                    if(!deadTets.empty())
                    {
                        tetId = deadTets.back();
                        deadTets.pop_back();
                        tets[tetId] = tet;
                        aliveTets[tetId] = true;
                    }
                    else
                    {
                        tetId = tets.size();
                        tets.push_back(tet);
                        aliveTets.push_back(true);
                    }

                    MeshNeigElem elem(MeshTet::ELEMENT_TYPE, tetId);
                    topos[tet.v[0]].neighborElems.push_back(elem);
                    topos[tet.v[1]].neighborElems.push_back(elem);
                    topos[tet.v[2]].neighborElems.push_back(elem);
                    topos[tet.v[3]].neighborElems.push_back(elem);
                }


                ++totalEdgeSwapCount;

                // Verify neighbor verts
                stillVertsToTry = true;
                for(const MeshNeigVert& vVert : vTopo.neighborVerts)
                    vertsToTry[vVert.v] = true;
                for(const MeshNeigVert& nVert : nTopo.neighborVerts)
                    vertsToTry[nVert.v] = true;
            }
        }
    }

    // Remove deleted tets
    size_t copyTetId = 0;
    for(size_t tId=0; tId < tets.size(); ++tId)
    {
        if(aliveTets[tId])
        {
            if(tId != copyTetId)
            {
                const MeshLocalTet& tet = tets[tId];
                for(uint i=0; i < 4; ++i)
                {
                    MeshTopo& topo = topos[tet.v[i]];
                    for(MeshNeigElem& ne : topo.neighborElems)
                        if(ne.id == tId) {ne.id = copyTetId; break;}
                }

                tets[copyTetId] = tet;
            }
            ++copyTetId;
        }
    }
    tets.resize(copyTetId);

    getLog().postMessage(new Message('I', false,
        "Edge swap:        " +
        std::to_string(passCount) + " passes \t(" +
        std::to_string(totalEdgeSwapCount) + " swap)",
        "BatrTopologist"));

//    for(auto rIt = edgeSwapCounters.begin();
//        rIt != edgeSwapCounters.end(); ++rIt)
//    {
//        getLog().postMessage(new Message('I', false,
//            "Ring of " + std::to_string(rIt->first) +
//            " verts count: " + std::to_string(rIt->second),
//            "BatrTopologist"));
//    }

    return totalEdgeSwapCount;
}

template<typename T, typename V>
bool BatrTopologist::popOut(std::vector<T>& vec, const V& val)
{
    size_t vecCount = vec.size();
    for(size_t i=0; i < vecCount; ++i)
    {
        if(vec[i] == val)
        {
            std::swap(vec[i], vec.back());
            vec.pop_back();
            return true;
        }
    }

    return false;
}

template<typename T, typename V>
bool BatrTopologist::popOut(std::vector<T>& vec,
                            const std::vector<V>& vals)
{
    size_t count = 0;

    size_t i=0;
    while(i < vec.size())
    {
        bool found = false;
        for(const V& val : vals)
        {
            if(vec[i] == val)
            {
                std::swap(vec[i], vec.back());
                vec.pop_back();
                found = true;
                ++count;
                break;
            }
        }

        if(!found)
            ++i;
    }

    return count == vals.size();
}

template<typename T, typename V>
bool BatrTopologist::make_union(std::vector<T>& vec, const V& val)
{
    size_t vecCount = vec.size();
    for(size_t i=0; i < vecCount; ++i)
    {
        if(vec[i] == val)
        {
            return false;
        }
    }

    vec.push_back(val);
    return true;
}

template<typename T>
void BatrTopologist::findRing(
        const std::vector<T> &tets,
        const std::vector<MeshTopo> &topos,
        uint vId, uint nId,
        std::vector<uint>& ringVerts,
        std::vector<uint>& ringElems) const
{
    const MeshTopo& vTopo = topos[vId];
    const MeshTopo& nTopo = topos[nId];
    const std::vector<MeshNeigElem>& vElems = vTopo.neighborElems;
    const std::vector<MeshNeigElem>& nElems = nTopo.neighborElems;

    for(const MeshNeigElem& vElem : vElems)
    {
        for(const MeshNeigElem& nElem : nElems)
        {
            if(vElem.id == nElem.id)
            {
                ringElems.push_back(vElem.id);
                break;
            }
        }
    }

    std::set<uint> ringSet;
    for(uint iElem : ringElems)
    {
        const MeshLocalTet& tet = tets[iElem];
        if(tet.v[0] != vId && tet.v[0] != nId) ringSet.insert(tet.v[0]);
        if(tet.v[1] != vId && tet.v[1] != nId) ringSet.insert(tet.v[1]);
        if(tet.v[2] != vId && tet.v[2] != nId) ringSet.insert(tet.v[2]);
        if(tet.v[3] != vId && tet.v[3] != nId) ringSet.insert(tet.v[3]);
    }

    ringVerts = std::vector<uint>(ringSet.begin(), ringSet.end());
}

void BatrTopologist::findExclusiveElems(
        const Mesh& mesh, uint vId,
        const std::vector<uint>& ringElems,
        std::vector<uint>& exElems) const
{
    const std::vector<MeshNeigElem>& vElems =
            mesh.topos[vId].neighborElems;

    for(const MeshNeigElem& vElem : vElems)
    {
        bool isIntersection = false;
        for(uint rElem : ringElems)
        {
            if(vElem.id == rElem)
            {
                isIntersection = true;
                break;
            }
        }

        if(!isIntersection)
            exElems.push_back(vElem.id);
    }
}

void BatrTopologist::findExclusiveVerts(
        const Mesh& mesh,
        uint vId, uint nId,
        const std::vector<uint>& ringVerts,
        std::vector<uint>& exVerts) const
{
    const std::vector<MeshNeigVert>& vVerts =
            mesh.topos[vId].neighborVerts;

    for(const MeshNeigVert& vVert : vVerts)
    {
        if(vVert.v == nId)
            continue;

        bool isIntersection = false;
        for(uint rElem : ringVerts)
        {
            if(vVert.v == rElem)
            {
                isIntersection = true;
                break;
            }
        }

        if(!isIntersection)
            exVerts.push_back(vVert.v);
    }
}
