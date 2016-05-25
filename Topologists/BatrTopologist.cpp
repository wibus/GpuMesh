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


const bool ENABLE_DEAD_REUSE = true;
const bool ENABLE_VERIFICATION_FRENZY = false;


inline MeshNeigElem toTet(uint eId)
{
    return MeshNeigElem(MeshTet::ELEMENT_TYPE, eId);
}

inline void appendElems(std::vector<MeshNeigElem>& elems, const std::vector<uint>& eIds)
{
    elems.reserve(elems.size() + eIds.size());
    for(uint eId : eIds) elems.push_back(toTet(eId));
}


BatrTopologist::BatrTopologist() :
    _refinementCoarseningMaxPassCount(10),
    _globalLoopMaxPassCount(10),
    _minAcceptableGenQuality(1e-9)
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
    if(mesh.tets.empty() || !(mesh.pris.empty() && mesh.hexs.empty()))
        return;

    getLog().postMessage(new Message('I', false,
        "Performing BATR topology modifications...",
        "BatrTopologist"));

    getLog().postMessage(new Message('I', false,
        "Start by removing lone verts and edges",
        "BatrTopologist"));

    std::vector<uint> vertsToVerify(mesh.verts.size());
    std::iota(vertsToVerify.begin(), vertsToVerify.end(), 0);
    std::vector<bool> aliveVerts(mesh.verts.size(), true);
    std::vector<bool> aliveTets(mesh.tets.size(), true);
    std::vector<uint> deadVerts, deadTets;
    cureBoundaries(
            mesh, vertsToVerify,
            aliveVerts, deadVerts,
            aliveTets, deadTets);
    trimTets(mesh, aliveTets);
    trimVerts(mesh, aliveVerts);

    getLog().postMessage(new Message('I', false,
        "And then start main modification loop",
        "BatrTopologist"));

    size_t passCount = 0;
    size_t lastPassOpCount = 0;
    while(true)
    {
        size_t passOpCount = 0;

        passOpCount += faceSwapping(mesh, crew);
        passOpCount += edgeSwapping(mesh, crew);
        passOpCount += edgeSplitMerge(mesh, crew);

        if(passOpCount == 0 ||
           passOpCount == lastPassOpCount ||
           passCount >= _globalLoopMaxPassCount)
        {
            break;
        }

        lastPassOpCount = passOpCount;
        ++passCount;
    }

    mesh.compileTopology(false);
}

void BatrTopologist::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{

}

size_t BatrTopologist::edgeSplitMerge(
        Mesh& mesh,
        const MeshCrew& crew) const
{
    std::vector<MeshTet>& tets = mesh.tets;
    std::vector<MeshVert>& verts = mesh.verts;
    std::vector<MeshTopo>& topos = mesh.topos;

    bool emergencyExit = false;
    bool stillVertsToTry = true;
    std::vector<bool> aliveVerts(verts.size(), true);
    std::vector<bool> vertsToTry(verts.size(), true);
    std::vector<bool> aliveTets(mesh.tets.size(), true);
    std::vector<uint> deadVerts;
    std::vector<uint> deadTets;

    size_t passCount = 0;
    size_t vertMergeCount = 0;
    size_t edgeSplitCount = 0;
    for(;passCount < _refinementCoarseningMaxPassCount && stillVertsToTry; ++passCount)
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
            findRing(mesh, vId, nId, ringVerts, ringElems);

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
                    double tetQual = crew.evaluator().tetQuality(mesh,
                            crew.sampler(), crew.measurer(), tets[dElem]);
                    if(tetQual <= _minAcceptableGenQuality)
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
                deadVerts.push_back(nId);
                aliveVerts[nId] = false;
                for(uint rElem : ringElems)
                {
                    deadTets.push_back(rElem);
                    aliveTets[rElem] = false;
                }

                // Replace n for v in n's exclusive elements
                for(uint ndElem : nExElems)
                {
                    MeshTet& tet = tets[ndElem];
                    if(tet.v[0] == nId)      tet.v[0] = vId;
                    else if(tet.v[1] == nId) tet.v[1] = vId;
                    else if(tet.v[2] == nId) tet.v[2] = vId;
                    else if(tet.v[3] == nId) tet.v[3] = vId;
                }

                // Remove n from ring verts
                // Remove ring elems from ring verts
                std::vector<uint> ringVertsCopy = ringVerts;
                for(uint rVert : ringVertsCopy)
                {
                    popOut(topos[rVert].neighborElems, ringElems);

                    // Kill vert if it relied entirely on ring elems
                    if(topos[rVert].neighborElems.empty())
                    {
                        popOut(ringVerts, rVert);
                        aliveVerts[rVert] = false;
                        deadVerts.push_back(rVert);
                    }
                }


                // Rebuild v elem neighborhood
                std::vector<MeshNeigElem>& vElems = vTopo.neighborElems;
                vElems.clear();
                appendElems(vElems, vExElems);
                appendElems(vElems, nExElems);
                vElems.shrink_to_fit();

                // Rebuild v vert neighborhood
                buildVertNeighborhood(mesh, vId);

                for(uint vVert : vTopo.neighborVerts)
                {
                    buildVertNeighborhood(mesh, vVert);
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


                std::vector<uint> vertsToVerify(
                    vTopo.neighborVerts.begin(),
                    vTopo.neighborVerts.end());
                vertsToVerify.push_back(vId);

                cureBoundaries(mesh, vertsToVerify,
                                aliveVerts, deadVerts,
                                aliveTets, deadTets);


                // Notify to check neighbor verts
                ++passVertMergeCount;
                stillVertsToTry = true;
                vertsToTry[vId] = true;
                vertsToTry[nId] = false;

                for(const MeshNeigVert& vVert : vTopo.neighborVerts)
                    vertsToTry[vVert.v] = true;

                if(ENABLE_VERIFICATION_FRENZY &&
                   !validateMesh(mesh, aliveTets, aliveVerts))
                {
                    for(MeshTet& tet : tets)
                        tet.value = 0.0;

                    for(const MeshNeigElem& vElem : topos[vId].neighborElems)
                        tets[vElem.id].value = 0.5;

                    emergencyExit = true;
                    break;
                }
            }
            else if(dist > maxEdgeLength())
            {
                // Splitting the edge
                MeshTopo wTopo;
                glm::dvec3 middle = (vert.p + neig.p) /2.0;
                if(ringElems.size() < ringVerts.size())
                {
                    wTopo.isBoundary = true;
                    wTopo.snapToBoundary = vTopo.snapToBoundary;
                    middle = (*vTopo.snapToBoundary)(middle);
                }


                uint wId;
                if(deadVerts.empty() || !ENABLE_DEAD_REUSE)
                {
                    wId = verts.size();

                    topos.push_back(wTopo);
                    verts.push_back(MeshVert(middle, vert.c));

                    aliveVerts.push_back(true);
                    vertsToTry.push_back(true);
                }
                else
                {
                    wId = deadVerts.back();
                    deadVerts.pop_back();

                    topos[wId] = wTopo;
                    verts[wId] = MeshVert(middle, vert.c);

                    aliveVerts[wId] = true;
                    vertsToTry[wId] = true;
                }


                bool isConformal = true;
                for(uint rElem : ringElems)
                {
                    const MeshTet& tet = tets[rElem];
                    MeshTet vTet = tet, nTet = tet;

                    if(tet.v[0] == vId) nTet.v[0] = wId;
                    else if(tet.v[0] == nId) vTet.v[0] = wId;
                    if(tet.v[1] == vId) nTet.v[1] = wId;
                    else if(tet.v[1] == nId) vTet.v[1] = wId;
                    if(tet.v[2] == vId) nTet.v[2] = wId;
                    else if(tet.v[2] == nId) vTet.v[2] = wId;
                    if(tet.v[3] == vId) nTet.v[3] = wId;
                    else if(tet.v[3] == nId) vTet.v[3] = wId;

                    double vTetQual = crew.evaluator().tetQuality(mesh,
                            crew.sampler(), crew.measurer(), vTet);
                    double nTetQual = crew.evaluator().tetQuality(mesh,
                            crew.sampler(), crew.measurer(), nTet);
                    if(vTetQual <= _minAcceptableGenQuality ||
                       nTetQual <= _minAcceptableGenQuality)
                    {
                        isConformal = false;
                        break;
                    }
                }

                if(!isConformal)
                {
                    // Rollback modifications
                    vertsToTry[wId] = false;
                    aliveVerts[wId] = false;
                    deadVerts.push_back(wId);
                    continue;
                }

                // Build new elements
                for(uint rElem : ringElems)
                {
                    MeshTet tet = tets[rElem];
                    popOut(topos[tet.v[0]].neighborElems, rElem);
                    popOut(topos[tet.v[1]].neighborElems, rElem);
                    popOut(topos[tet.v[2]].neighborElems, rElem);
                    popOut(topos[tet.v[3]].neighborElems, rElem);
                    deadTets.push_back(rElem);
                    aliveTets[rElem] = false;

                    MeshTet vTet = tet;
                    MeshTet nTet = tet;

                    if(tet.v[0] == vId) nTet.v[0] = wId;
                    else if(tet.v[0] == nId) vTet.v[0] = wId;
                    if(tet.v[1] == vId) nTet.v[1] = wId;
                    else if(tet.v[1] == nId) vTet.v[1] = wId;
                    if(tet.v[2] == vId) nTet.v[2] = wId;
                    else if(tet.v[2] == nId) vTet.v[2] = wId;
                    if(tet.v[3] == vId) nTet.v[3] = wId;
                    else if(tet.v[3] == nId) vTet.v[3] = wId;


                    if(AbstractMeasurer::tetEuclideanVolume(mesh, vTet) < 0.0)
                        std::swap(vTet.v[0], vTet.v[1]);

                    if(AbstractMeasurer::tetEuclideanVolume(mesh, nTet) < 0.0)
                        std::swap(nTet.v[0], nTet.v[1]);


                    MeshNeigElem vElem;
                    if(deadTets.empty() || !ENABLE_DEAD_REUSE)
                    {
                        vElem = toTet(tets.size());
                        aliveTets.push_back(true);
                        tets.push_back(vTet);
                    }
                    else
                    {
                        vElem = toTet(deadTets.back());
                        deadTets.pop_back();

                        aliveTets[vElem] = true;
                        tets[vElem] = vTet;
                    }

                    MeshNeigElem nElem;
                    if(deadTets.empty() || !ENABLE_DEAD_REUSE)
                    {
                        nElem = toTet(tets.size());
                        aliveTets.push_back(true);
                        tets.push_back(nTet);
                    }
                    else
                    {
                        nElem = toTet(deadTets.back());
                        deadTets.pop_back();

                        aliveTets[nElem] = true;
                        tets[nElem] = nTet;
                    }

                    topos[vTet.v[0]].neighborElems.push_back(vElem);
                    topos[vTet.v[1]].neighborElems.push_back(vElem);
                    topos[vTet.v[2]].neighborElems.push_back(vElem);
                    topos[vTet.v[3]].neighborElems.push_back(vElem);

                    topos[nTet.v[0]].neighborElems.push_back(nElem);
                    topos[nTet.v[1]].neighborElems.push_back(nElem);
                    topos[nTet.v[2]].neighborElems.push_back(nElem);
                    topos[nTet.v[3]].neighborElems.push_back(nElem);
                }


                // Connect w to ring verts
                buildVertNeighborhood(mesh, wId);
                for(const MeshNeigVert& wVert : topos[wId].neighborVerts)
                {
                    buildVertNeighborhood(mesh, wVert);
                    vertsToTry[wVert.v] = true;
                }



                std::vector<uint> vertsToVerify(
                    wTopo.neighborVerts.begin(),
                    wTopo.neighborVerts.end());
                vertsToVerify.push_back(wId);

                cureBoundaries(mesh, vertsToVerify,
                                aliveVerts, deadVerts,
                                aliveTets, deadTets);


                // Notify to check neighbor verts
                ++passEdgeSplitCount;
                stillVertsToTry = true;

                if(ENABLE_VERIFICATION_FRENZY &&
                   !validateMesh(mesh, aliveTets, aliveVerts))
                {
                    for(MeshTet& tet : tets)
                        tet.value = 0.0;

                    for(const MeshNeigElem& vElem : topos[vId].neighborElems)
                        tets[vElem.id].value = 0.5;

                    for(const MeshNeigElem& nElem : topos[nId].neighborElems)
                        tets[nElem.id].value = 1.0;

                    emergencyExit = true;
                    break;
                }
            }
        }

        vertMergeCount += passVertMergeCount;
        edgeSplitCount += passEdgeSplitCount;

        if(emergencyExit)
            break;
    }

    validateMesh(mesh, aliveTets, aliveVerts);

    trimTets(mesh, aliveTets);
    trimVerts(mesh, aliveVerts);


    getLog().postMessage(new Message('I', false,
        "Edge split/merge: " +
        std::to_string(passCount) + " passes \t(" +
        std::to_string(edgeSplitCount) + " split, " +
        std::to_string(vertMergeCount) + " merge)",
        "BatrTopologist"));


    if(!emergencyExit)
        return edgeSplitCount + vertMergeCount;
    else
        return 0;
}

size_t BatrTopologist::faceSwapping(
        Mesh& mesh,
        const MeshCrew& crew) const
{
    std::vector<MeshTet>& tets = mesh.tets;
    std::vector<MeshTopo>& topos = mesh.topos;

    size_t passCount = 0;
    size_t faceSwapCount = 0;
    bool stillTetsToTry = true;
    bool emergencyExit = false;

    std::vector<bool> aliveVerts(mesh.verts.size(), true);
    std::vector<bool> aliveTets(tets.size(), true);
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
                            topos[tOp].neighborVerts.push_back(MeshNeigVert(nOp));

                            topos[nOp].neighborElems.push_back(MeshNeigElem(MeshTet::ELEMENT_TYPE, t));
                            topos[nOp].neighborElems.push_back(MeshNeigElem(MeshTet::ELEMENT_TYPE, lt));
                            topos[nOp].neighborVerts.push_back(MeshNeigVert(tOp));

                            for(MeshNeigElem& v : neigTets0)
                                if(v.id == nt) {v.id = lt; break;}

                            for(MeshNeigElem& v : neigTets2)
                                if(v.id == t) {v.id = lt; break;}

                            // TODO wbussiere 2016-04-20 :
                            // Expand 'to try' marker to neighboor tets
                            stillTetsToTry = true;
                            tetsToTry.push_back(true);
                            aliveTets.push_back(true);
                            tetsToTry[nt] = true;
                            tetsToTry[t] = true;
                            ++faceSwapCount;


                            if(ENABLE_VERIFICATION_FRENZY &&
                               !validateMesh(mesh, aliveTets, aliveVerts))
                            {
                                for(MeshTet& tet : tets)
                                    tet.value = 0.0;

                                for(const MeshNeigElem& vElem : topos[tOp].neighborElems)
                                    tets[vElem.id].value = 0.5;

                                for(const MeshNeigElem& nElem : topos[nOp].neighborElems)
                                    tets[nElem.id].value = 1.0;

                                emergencyExit = true;
                                break;
                            }

                            break;
                        }
                    }
                }
            }

            if(emergencyExit)
                break;
        }

        if(emergencyExit)
            break;
    }

    getLog().postMessage(new Message('I', false,
        "Face swap:        " +
        std::to_string(passCount) + " passes \t(" +
        std::to_string(faceSwapCount) + " swap)",
        "BatrTopologist"));

    if(!emergencyExit)
        return faceSwapCount;
    else
        return 0;
}

size_t BatrTopologist::edgeSwapping(
        Mesh& mesh,
        const MeshCrew& crew) const
{
    std::vector<MeshTet>& tets = mesh.tets;
    std::vector<MeshVert>& verts = mesh.verts;
    std::vector<MeshTopo>& topos = mesh.topos;

    bool emergencyExit = false;
    bool stillVertsToTry = true;
    std::vector<bool> vertsToTry(verts.size(), true);
    std::vector<bool> aliveVerts(verts.size(), true);
    std::vector<bool> aliveTets(tets.size(), true);
    std::vector<uint> deadVerts;
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
                findRing(mesh, vId, nId, ringVerts, ringElems);

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
                            printRing(mesh, vId, nId);
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

                // Debug output for unclosable tet rings
                if(nIt != ringVerts.end())
                {
                    std::vector<uint> elems(vTopo.neighborElems.begin(), vTopo.neighborElems.end());
                    std::vector<uint> nElems(topos[nId].neighborElems.begin(), topos[nId].neighborElems.end());
                    popOut(elems, nElems);
                    elems.insert(elems.end(), nElems.begin(), nElems.end());
                    popOut(elems, ringElems);


                    for(MeshTet& tet : tets)
                    {
                        tet.value = 1.0;
                    }

                    for(uint& elem : elems)
                    {
                        tets[elem].value = 0.0;
                    }

                    for(uint rElem : ringElems)
                    {
                        tets[rElem].value = 0.5;
                    }

                    trimTets(mesh, aliveTets);
                    return 0;
                }


                // Verify that that this ring size can be handled
                size_t ringSize = ringVerts.size();
                ++ringSizeCounters[ringSize];
                if(ringSize < 3 || ringSize >= _ringConfigDictionary.size())
                {
                    ++edgeSwapCounters[-ringSize];
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
                        outsidersOk = false;
                        break;

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
                    ++totalEdgeSwapCount;
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
                    if(deadTets.empty() || !ENABLE_DEAD_REUSE)
                    {
                        tetId = tets.size();
                        tets.push_back(tet);
                        aliveTets.push_back(true);
                    }
                    else
                    {
                        tetId = deadTets.back();
                        deadTets.pop_back();
                        tets[tetId] = tet;
                        aliveTets[tetId] = true;
                    }

                    MeshNeigElem elem(MeshTet::ELEMENT_TYPE, tetId);
                    topos[tet.v[0]].neighborElems.push_back(elem);
                    topos[tet.v[1]].neighborElems.push_back(elem);
                    topos[tet.v[2]].neighborElems.push_back(elem);
                    topos[tet.v[3]].neighborElems.push_back(elem);
                }


                // Verify neighbor verts
                stillVertsToTry = true;
                for(const MeshNeigVert& vVert : vTopo.neighborVerts)
                    vertsToTry[vVert.v] = true;
                for(const MeshNeigVert& nVert : nTopo.neighborVerts)
                    vertsToTry[nVert.v] = true;


                std::vector<uint> vertsToVerify = ringVerts;
                vertsToVerify.push_back(vId);
                vertsToVerify.push_back(nId);

                cureBoundaries(mesh, vertsToVerify,
                                aliveVerts, deadVerts,
                                aliveTets, deadTets);

                if(ENABLE_VERIFICATION_FRENZY &&
                   !validateMesh(mesh, aliveTets, aliveVerts))
                {
                    for(MeshTet& tet : tets)
                        tet.value = 0.0;

                    for(const MeshNeigElem& vElem : vElems)
                        tets[vElem.id].value = 0.5;

                    for(const MeshNeigElem& nElem : nElems)
                        tets[nElem.id].value = 1.0;

                    emergencyExit = true;
                    break;
                }
            }

            if(emergencyExit)
                break;
        }

        if(emergencyExit)
            break;
    }

    trimTets(mesh, aliveTets);
    trimVerts(mesh, aliveVerts);

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

    if(!emergencyExit)
        return totalEdgeSwapCount;
    else
        return 0;
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
    size_t popCount = 0;

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
                ++popCount;
                break;
            }
        }

        if(!found)
            ++i;
    }

    return popCount == vals.size();
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

void BatrTopologist::buildVertNeighborhood(Mesh& mesh, uint vId) const
{
    std::set<uint> neigSet;
    MeshTopo& vTopo = mesh.topos[vId];
    for(const MeshNeigElem& vElem : vTopo.neighborElems)
    {
        const MeshTet& tet = mesh.tets[vElem];
        if(tet.v[0] != vId) neigSet.insert(tet.v[0]);
        if(tet.v[1] != vId) neigSet.insert(tet.v[1]);
        if(tet.v[2] != vId) neigSet.insert(tet.v[2]);
        if(tet.v[3] != vId) neigSet.insert(tet.v[3]);
    }

    vTopo.neighborVerts = std::vector<MeshNeigVert>(
        neigSet.begin(), neigSet.end());
}

void BatrTopologist::findRing(
        const Mesh &mesh,
        uint vId, uint nId,
        std::vector<uint>& ringVerts,
        std::vector<uint>& ringElems) const
{
    const MeshTopo& vTopo = mesh.topos[vId];
    const MeshTopo& nTopo = mesh.topos[nId];
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
        const MeshTet& tet = mesh.tets[iElem];
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

void BatrTopologist::trimVerts(Mesh& mesh, const std::vector<bool>& aliveVerts) const
{
    std::vector<MeshVert>& verts = mesh.verts;
    std::vector<MeshTopo>& topos = mesh.topos;

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
                    MeshTet& tet = mesh.tets[ne.id];
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
}

void BatrTopologist::trimTets(Mesh& mesh, const std::vector<bool>& aliveTets) const
{
    std::vector<MeshTopo>& topos = mesh.topos;
    std::vector<MeshTet>& tets = mesh.tets;

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
}

bool BatrTopologist::cureBoundaries(
        Mesh& mesh,
        std::vector<uint>& vertsToVerifiy,
        std::vector<bool>& aliveVerts,
        std::vector<uint>& deadVerts,
        std::vector<bool>& aliveElems,
        std::vector<uint>& deadElems) const
{
    bool meshTouched = false;
    std::vector<MeshTopo>& topos = mesh.topos;

    for(uint vArId = 0; vArId < vertsToVerifiy.size(); ++vArId)
    {
        uint vId = vertsToVerifiy[vArId];
        const MeshTopo& vTopo = topos[vId];

        if(!aliveVerts[vId])
        {
            continue;
        }

        // Check for lone verts (verts that lives on a single tet)
        if(vTopo.neighborElems.size() == 1)
        {
            uint vElem = vTopo.neighborElems.front();
            for(const MeshNeigVert& vVert : vTopo.neighborVerts)
            {
                popOut(topos[vVert].neighborElems, vElem);
                buildVertNeighborhood(mesh, vVert.v);
                vertsToVerifiy.push_back(vVert);
            }

            aliveElems[vElem] = false;
            deadElems.push_back(vElem);

            aliveVerts[vId] = false;
            deadVerts.push_back(vId);
            meshTouched = true;

//            getLog().postMessage(new Message('D', false,
//                "Removing lone vert " + std::to_string(vId) +
//                " living on tet " + std::to_string(vElem),
//                "BatrTopologist"));

            // This not a valid vert anymore
            continue;
        }


        // Check for lone edges (edges that lives on a single tet)
        for(const MeshNeigVert& vVert : vTopo.neighborVerts)
        {
            uint nId = vVert.v;
            const MeshTopo& nTopo = topos[nId];

            std::vector<uint> ringElems;
            for(const MeshNeigElem& vElem : vTopo.neighborElems)
            {
                for(const MeshNeigElem& nElem : nTopo.neighborElems)
                {
                    if(vElem.id == nElem.id)
                    {
                        ringElems.push_back(vElem.id);
                        break;
                    }
                }
            }

            if(ringElems.size() == 1)
            {
                uint tId = ringElems.front();
                const MeshTet& tet = mesh.tets[tId];
                if(!topos[tet.v[0]].isBoundary || !topos[tet.v[1]].isBoundary ||
                   !topos[tet.v[2]].isBoundary || !topos[tet.v[3]].isBoundary)
                {
//                    getLog().postMessage(new Message('D', false,
//                        "Skipping lone edge " + std::to_string(vId) + "-" + std::to_string(nId)
//                        + " living on tet " + std::to_string(tId) + " not on boundary",
//                        "BatrTopologist"));
                    continue;
                }

                for(int i=0; i < 4; ++i)
                {
                    popOut(topos[tet.v[i]].neighborElems, tId);
                    buildVertNeighborhood(mesh, tet.v[i]);
                    vertsToVerifiy.push_back(tet.v[i]);
                }

                aliveElems[tId] = false;
                deadElems.push_back(tId);
                meshTouched = true;

//                getLog().postMessage(new Message('D', false,
//                    "Removing lone edge " + std::to_string(vId) + "-" + std::to_string(nId)
//                    + " living on tet " + std::to_string(tId),
//                    "BatrTopologist"));

                // Vert neighborhood was invalidated, but this
                // vert was pushed back in vertsToVerify.
                break;
            }
        }
    }

    if(meshTouched && ENABLE_VERIFICATION_FRENZY)
    {
        if(!validateMesh(mesh, aliveElems, aliveVerts))
        {
            getLog().postMessage(new Message('E', false,
                "Removing lone vert or edge destroyed mesh conformity...",
                "BatrTopologist"));
        }
    }

    return meshTouched;
}

std::ostream& print(const Mesh& mesh, uint vId)
{
    std::cout << vId;
    if(mesh.topos[vId].isBoundary)
        std::cout << "*";
    return std::cout;
}

void BatrTopologist::printRing(
    const Mesh& mesh, uint vId, uint nId) const
{
    std::vector<uint> ringVerts;
    std::vector<uint> ringElems;
    findRing(mesh, vId, nId, ringVerts, ringElems);
    std::sort(ringVerts.begin(), ringVerts.end());
    std::sort(ringElems.begin(), ringElems.end());

    std::vector<uint> vCount(ringVerts.size(), 0);
    for(size_t i=0; i < ringElems.size(); ++i)
    {
        const MeshTet& tet = mesh.tets[ringElems[i]];

        for(size_t t=0; t < 4; ++t)
            for(size_t v=0; v < ringVerts.size(); ++v)
                if(ringVerts[v] == tet.v[t])
                    {++vCount[v]; break;}
    }

    std::cout << "Ring (";
    print(mesh, vId);
    std::cout << ", ";
    print(mesh, nId);
    std::cout << ")" << std::endl;

    std::cout << "Verts ( " << ringVerts.size() << ") : [";
    for(size_t i=0; i < ringVerts.size(); ++i)
    {
        print(mesh, ringVerts[i]) << "(" << vCount[i] << ")";
        if(i < ringVerts.size()-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;


    std::cout << "Elems (" << ringElems.size() << ") : [";
    for(size_t i=0; i < ringElems.size(); ++i)
    {
        std::cout << ringElems[i];
        if(i < ringElems.size()-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Tets : [";
    for(size_t i=0; i < ringElems.size(); ++i)
    {
        const MeshTet& tet = mesh.tets[ringElems[i]];
        std::cout << "(";
        print(mesh, tet.v[0]) << ", ";
        print(mesh, tet.v[1]) << ", ";
        print(mesh, tet.v[2]) << ", ";
        print(mesh, tet.v[3]) << ") ";
    }
    std::cout << "]" << std::endl;
}

bool BatrTopologist::validateMesh(const Mesh& mesh,
                  const std::vector<bool>& aliveTets,
                  const std::vector<bool>& aliveVerts) const
{
    const std::vector<MeshVert>& verts = mesh.verts;
    const std::vector<MeshTopo>& topos = mesh.topos;
    const std::vector<MeshTet>& tets = mesh.tets;

    if(verts.size() != topos.size())
    {
        getLog().postMessage(new Message('E', false,
            "verts and topos size mismatch", "BatrTopologist"));
        return false;
    }

    if(verts.size() != aliveVerts.size())
    {
        getLog().postMessage(new Message('E', false,
            "verts and aliveVerts size mismatch", "BatrTopologist"));
        return false;
    }

    if(tets.size() != aliveTets.size())
    {
        getLog().postMessage(new Message('E', false,
            "tets and aliveTets size mismatch", "BatrTopologist"));
        return false;
    }


    for(size_t tId=0; tId < tets.size(); ++tId)
    {
        if(!aliveTets[tId])
            continue;

        const MeshTet& tet = tets[tId];

        for(size_t i=0; i < 4; ++i)
        {
            if(!aliveVerts[tet.v[i]])
            {
                getLog().postMessage(new Message('E', false,
                    "tet referencing dead vert " + std::to_string(tet.v[i]),
                    "BatrTopologist"));
                return false;
            }

            bool refFound = false;
            const MeshTopo& topo = topos[tet.v[i]];
            for(const MeshNeigElem& elem : topo.neighborElems)
            {
                if(elem.id == tId)
                {
                    refFound = true;
                    break;
                }
            }

            if(!refFound)
            {
                getLog().postMessage(new Message('E', false,
                    "tet(" + std::to_string(tId) + ") referencing vert" +
                    std::to_string(tet.v[i])+ " that doesn't reference tet back",
                    "BatrTopologist"));
                return false;
            }
        }

    }


    for(size_t vId=0; vId < verts.size(); ++vId)
    {
        if(!aliveVerts[vId])
            continue;

        std::set<uint> neighSet;
        for(const MeshNeigElem& aElem : topos[vId].neighborElems)
        {
            if(!aliveTets[aElem.id])
            {
                getLog().postMessage(new Message('E', false,
                    "vert referencing dead tet " + std::to_string(aElem.id),
                    "BatrTopologist"));
                return false;
            }

            MeshTet tet = tets[aElem.id];
            if(tet.v[0] != vId) neighSet.insert(tet.v[0]);
            if(tet.v[1] != vId) neighSet.insert(tet.v[1]);
            if(tet.v[2] != vId) neighSet.insert(tet.v[2]);
            if(tet.v[3] != vId) neighSet.insert(tet.v[3]);

            if(tet.v[0] != vId && tet.v[1] != vId &&
               tet.v[2] != vId && tet.v[3] != vId)
            {
                getLog().postMessage(new Message('E', false,
                    "vert(" + std::to_string(vId) + ") referencing tet(" +
                    std::to_string(aElem.id)+ ") that doesn't reference vert back",
                    "BatrTopologist"));
                return false;
            }
        }

        std::vector<uint> neighVec(neighSet.begin(), neighSet.end());
        std::vector<MeshNeigVert> aVerts = topos[vId].neighborVerts;
        std::sort(aVerts.begin(), aVerts.end());

        if(neighVec.size() == aVerts.size())
        {
            size_t count = neighVec.size();
            for(size_t i=0; i < count; ++i)
            {
                if(!aliveVerts[aVerts[i].v])
                {
                    getLog().postMessage(new Message('E', false,
                        "Vert referencing dead vert " + std::to_string(aVerts[i].v),
                        "BatrTopologist"));
                    return false;
                }

                if(neighVec[i] != aVerts[i].v)
                {
                    getLog().postMessage(new Message('E', false,
                        "Explicit and implicit vert(" + std::to_string(vId) +
                        ") neighborhood content mismatch", "BatrTopologist"));

                    return false;
                }

                bool refFound = false;
                for(const MeshNeigVert& vert : topos[aVerts[i].v].neighborVerts)
                {
                    if(vert.v == vId)
                    {
                        refFound = true;
                        break;
                    }
                }

                if(!refFound)
                {
                    getLog().postMessage(new Message('E', false,
                        "vert(" + std::to_string(vId) + ") referencing vert(" +
                        std::to_string(aVerts[i].v) + ") that doesn't reference vert back",
                        "BatrTopologist"));
                    return false;
                }
            }
        }
        else
        {
            getLog().postMessage(new Message('E', false,
                "Explicit and implicit vert(" + std::to_string(vId) +
                ") neighborhood size mismatch", "BatrTopologist"));

            return false;
        }
    }

    for(size_t vId=0; vId < verts.size(); ++vId)
    {
        if(!aliveVerts[vId])
            continue;

        for(uint nId : topos[vId].neighborVerts)
        {
            if(nId < vId)
            {
                continue;
            }

            std::vector<uint> ringVerts;
            std::vector<uint> ringElems;
            findRing(mesh, vId, nId, ringVerts, ringElems);

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
                        getLog().postMessage(new Message('E', false,
                            "Cannot close tet ring("+ std::to_string(vId) +"-"+
                            std::to_string(nId) +")", "BatrTopologist"));
                        printRing(mesh, vId, nId);
                        return false;
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
            {
                getLog().postMessage(new Message('E', false,
                    "Cannot close tet ring("+ std::to_string(vId) +"-"+
                    std::to_string(nId) +")", "BatrTopologist"));
                printRing(mesh, vId, nId);
                return false;
            }
        }
    }

    return true;
}
