#include "Mesh.h"

#include <map>
#include <iostream>

using namespace std;



std::vector<TetListNode*> TetList::_nodePool;
std::vector<Tetrahedron*> Mesh::_tetPool;

void Mesh::initialize(
        const std::vector<glm::dvec3>& boundingVertices,
        const std::vector<Tetrahedron>& boundingTetrahedras)
{
    vert.resize(boundingVertices.size());
    externalVertCount = boundingVertices.size();
    for(int i=0; i<vert.size(); ++i)
    {
        vert[i] = Vertex(boundingVertices[i]);
    }

    tetra.clear();
    for(auto& t : boundingTetrahedras)
    {
        Tetrahedron* tet = new Tetrahedron(t);
        tetra.push_back(tet);
    }
}

void Mesh::compileFacesAttributes(
        const glm::dvec4& cutPlaneEq,
        std::vector<glm::dvec3>& vertices,
        std::vector<glm::dvec3>& normals,
        std::vector<glm::dvec3>& triEdges,
        std::vector<double>& tetQualities)
{
    vertices.reserve(elemCount());
    normals.reserve(elemCount());
    triEdges.reserve(elemCount());
    tetQualities.reserve(elemCount());

    qualityCount = 0;
    qualityMean = 0;
    qualityVar = 0;

    glm::dvec3 cutNormal(cutPlaneEq);
    double cutDistance = cutPlaneEq.w;

    for(const auto& tet : tetra)
    {
        if(isExternalTetraHedron(tet))
            continue;

        glm::dvec3 verts[] = {
            vert[tet->v[0]].p,
            vert[tet->v[1]].p,
            vert[tet->v[2]].p,
            vert[tet->v[3]].p
        };

        if(glm::dot(verts[0], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[1], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[2], cutNormal) - cutDistance > 0.0 ||
           glm::dot(verts[3], cutNormal) - cutDistance > 0.0)
            continue;

        glm::dvec3 norms[] = {
            glm::normalize(glm::cross(verts[1] - verts[0], verts[2] - verts[1])),
            glm::normalize(glm::cross(verts[2] - verts[0], verts[3] - verts[2])),
            glm::normalize(glm::cross(verts[3] - verts[0], verts[1] - verts[3])),
            glm::normalize(glm::cross(verts[3] - verts[1], verts[2] - verts[3])),
        };

        double quality = tetrahedronQuality(tet);

        pushTriangle(vertices, normals, triEdges, tetQualities,
                     verts[0], verts[1], verts[2], norms[0], quality);
        pushTriangle(vertices, normals, triEdges, tetQualities,
                     verts[0], verts[2], verts[3], norms[1], quality);
        pushTriangle(vertices, normals, triEdges, tetQualities,
                     verts[0], verts[3], verts[1], norms[2], quality);
        pushTriangle(vertices, normals, triEdges, tetQualities,
                     verts[1], verts[3], verts[2], norms[3], quality);


        // Quality statistics
        qualityMean = (qualityMean * qualityCount + quality) /
                      (qualityCount + 1);
        double qualityMeanDist = qualityMean - quality;
        double qualityMeanDist2 = qualityMeanDist*qualityMeanDist;
        qualityVar = (qualityVar * qualityCount + qualityMeanDist2) /
                     (qualityCount + 1);
        ++qualityCount;
    }
}

bool Mesh::isExternalTetraHedron(Tetrahedron* tet)
{
    return tet->v[0] < externalVertCount ||
           tet->v[1] < externalVertCount ||
           tet->v[2] < externalVertCount ||
           tet->v[3] < externalVertCount;
}

double Mesh::tetrahedronQuality(Tetrahedron* tet)
{
    glm::dvec3& A = vert[tet->v[0]].p;
    glm::dvec3& B = vert[tet->v[1]].p;
    glm::dvec3& C = vert[tet->v[2]].p;
    glm::dvec3& D = vert[tet->v[3]].p;
    std::vector<double> lengths {
        glm::distance(A, B),
        glm::distance(A, C),
        glm::distance(A, D),
        glm::distance(B, C),
        glm::distance(D, B),
        glm::distance(C, D)
    };

    double maxLen = 0;
    for(auto l : lengths)
        if(l > maxLen)
            maxLen = l;

    double u = lengths[0];
    double v = lengths[1];
    double w = lengths[2];
    double U = lengths[5];
    double V = lengths[4];
    double W = lengths[3];

    double Volume = 4*u*u*v*v*w*w;
    Volume -= u*u*pow(v*v+w*w-U*U,2);
    Volume -= v*v*pow(w*w+u*u-V*V,2);
    Volume -= w*w*pow(u*u+v*v-W*W,2);
    Volume += (v*v+w*w-U*U)*(w*w+u*u-V*V)*(u*u+v*v-W*W);
    Volume = sqrt(Volume);
    Volume /= 12;

    double s1 = (double) ((U + V + W) / 2);
    double s2 = (double) ((u + v + W) / 2);
    double s3 = (double) ((u + V + w) / 2);
    double s4 = (double) ((U + v + w) / 2);

    double L1 = sqrt(s1*(s1-U)*(s1-V)*(s1-W));
    double L2 = sqrt(s2*(s2-u)*(s2-v)*(s2-W));
    double L3 = sqrt(s3*(s3-u)*(s3-V)*(s3-w));
    double L4 = sqrt(s4*(s4-U)*(s4-v)*(s4-w));

    double R = (Volume*3)/(L1+L2+L3+L4);

    return (4.89897948557) * R / maxLen;
}

void Mesh::pushTriangle(
        std::vector<glm::dvec3>& vertices,
        std::vector<glm::dvec3>& normals,
        std::vector<glm::dvec3>& triEdges,
        std::vector<double>& tetQualities,
        const glm::dvec3& A,
        const glm::dvec3& B,
        const glm::dvec3& C,
        const glm::dvec3& n,
        double quality)
{
    const glm::dvec3 X_EDGE(1, 1, 0);
    const glm::dvec3 Y_EDGE(0, 1, 1);
    const glm::dvec3 Z_EDGE(1, 0, 1);

    vertices.push_back(A);
    normals.push_back(n);
    triEdges.push_back(X_EDGE);
    tetQualities.push_back(quality);

    vertices.push_back(B);
    normals.push_back(n);
    triEdges.push_back(Y_EDGE);
    tetQualities.push_back(quality);

    vertices.push_back(C);
    normals.push_back(n);
    triEdges.push_back(Z_EDGE);
    tetQualities.push_back(quality);
}

bool Mesh::intersects(const glm::dvec3& v, Tetrahedron* tet)
{
    glm::dvec3 dist = v - tet->circumCenter;
    if(glm::dot(dist, dist) < tet->circumRadius2)
        return true;

    return false;
}

void Mesh::insertVertices(const std::vector<glm::dvec3>& vertices)
{
    int idStart = vert.size();
    int idEnd = idStart + vertices.size();
    vert.resize(idEnd);

    for(int i=idStart; i<idEnd; ++i)
    {
        const glm::dvec3& pos = vertices[i-idStart];
        cMin = glm::min(cMin, pos);
        cMax = glm::max(cMax, pos);
        vert[i] = Vertex(pos);
    }

    glm::dvec3 dim = cMax - cMin;
    cExtMin = cMin - dim / 1000000.0;
    cExtMax = cMax + dim / 1000000.0;
    cExtDim = (cExtMax - cExtMin);

    std::cout << "Initializing grid" << endl;
    initializeGrid(idStart, idEnd);
    int cellCount = gridSize.x*gridSize.y*gridSize.z;


    std::cout << "Inserting vertices in the mesh" << endl;
    int cId = 0;
    for(int k=0; k<gridSize.z; ++k)
    {
        for(int j=0; j<gridSize.y; ++j)
        {
            for(int i=0; i<gridSize.x; ++i, ++cId)
            {
                insertCell(glm::ivec3(i, j, k));
            }
        }

        double progress = (cId) / (double) (cellCount);
        std::cout <<  progress * 100 << "% done" << endl;
    }


    std::cout << "Collecting tetrahedrons" << endl;
    tearDownGrid();
}

void Mesh::initializeGrid(int idStart, int idEnd)
{
    // Compute dimensions
    const int VERT_PER_CELL = 5;
    int vCount = idEnd - idStart;
    int sideLen = (int) glm::pow(vCount / VERT_PER_CELL, 1/3.0);
    int height = vCount / (sideLen * sideLen) / VERT_PER_CELL;
    gridSize = glm::ivec3(sideLen, sideLen, height);
    int cellCount = gridSize.x*gridSize.y*gridSize.z;
    std::cout << "Grid size: "
              << gridSize.x << "x"
              << gridSize.y << "x"
              << gridSize.z << endl;
    std::cout << "Cell density: vert count/cell count = " <<
                 (idEnd-idStart) / (double) cellCount << endl;

    // Reset visit time
    _currentVisitTime = 0;


    // Construct grid
    grid.resize(gridSize.z);
    for(int k=0; k<gridSize.z; ++k)
    {
        grid[k].resize(gridSize.y);
        for(int j=0; j<gridSize.y; ++j)
        {
            grid[k][j].resize(gridSize.x);
        }
    }

    // Bin the vertices
    glm::dvec3 floatSize = glm::dvec3(gridSize);
    for(int vId=0; vId<idStart; ++vId)
    {
        vert[vId].visitTime = _currentVisitTime;

        const glm::dvec3& v = glm::clamp(vert[vId].p, cMin, cMax);
        glm::ivec3 bin = glm::ivec3((v - cExtMin) / (cExtDim) * floatSize);
        grid[bin.z][bin.y][bin.x].insertedVertId.push_back(vId);
    }
    for(int vId=idStart; vId<idEnd; ++vId)
    {
        vert[vId].visitTime = _currentVisitTime;

        const glm::dvec3& v = vert[vId].p;
        glm::ivec3 bin = glm::ivec3((v - cExtMin) / (cExtDim) * floatSize);
        grid[bin.z][bin.y][bin.x].waitingVertId.push_back(vId);
    }

    // Put starting tetrahedrons in the first cell
    for(auto tet : tetra)
    {
        insertTetrahedronGrid(
            tet->v[0],
            tet->v[1],
            tet->v[2],
            tet->v[3]);

        delete tet;
    }
    tetra.clear();

    //* Radially sort cell vertices
    for(int k=0; k<gridSize.z; ++k)
    {
        for(int j=0; j<gridSize.y; ++j)
        {
            for(int i=0; i<gridSize.x; ++i)
            {
                glm::dvec3 floatBin = glm::dvec3(i+1, j+1, k+1) / floatSize;
                glm::dvec3 cellCorner = floatBin * cExtDim + cMin;

                std::sort(grid[k][j][i].waitingVertId.begin(),
                          grid[k][j][i].waitingVertId.end(),
                          [this, &cellCorner](int a, int b) {
                    glm::dvec3 distA = cellCorner - vert[a].p;
                    glm::dvec3 distB = cellCorner - vert[b].p;
                    return glm::dot(distA, distA) < glm::dot(distB, distB);
                });
            }
        }
    }
    //*/
}


void Mesh::insertCell(const glm::ivec3& cId)
{
    GridCell& cell = grid[cId.z][cId.y][cId.x];
    std::vector<int>& waitingVertId = cell.waitingVertId;
    std::vector<int>& insertedVertId = cell.insertedVertId;
    insertedVertId.reserve(insertedVertId.size() + waitingVertId.size());

    int vertCount = waitingVertId.size();
    for(int vId=0; vId<vertCount; ++vId)
    {
        insertVertexGrid(cId, waitingVertId[vId]);
        insertedVertId.push_back(waitingVertId[vId]);
    }

    // Freeing space
    waitingVertId.clear();
    waitingVertId.shrink_to_fit();
}

void Mesh::insertVertexGrid(const glm::ivec3& cId, int vId)
{
    std::unordered_set<Triangle> ball;
    Tetrahedron* base = findBaseTetrahedron(cId, vId);
    findDelaunayBall(vId, base, ball);
    remeshDelaunayBall(cId, vId, ball);
}

Tetrahedron* Mesh::findBaseTetrahedron(const glm::ivec3& cId, int vId)
{
    const glm::dvec3& v = vert[vId].p;

    _baseQueue.clear();
    ++_currentVisitTime;

    _baseQueue.push_back(make_pair(cId, STATIC));
    for(int qId = 0; qId < _baseQueue.size(); ++qId)
    {
        glm::ivec3& c = _baseQueue[qId].first;
        const EDir& dir = _baseQueue[qId].second;
        c += DIR[dir];

        if(0 <= c.x && c.x < gridSize.x &&
           0 <= c.y && c.y < gridSize.y)
        {
            GridCell& cell = grid[c.z][c.y][c.x];
            std::vector<int>& insertedVertId = cell.insertedVertId;

            int insertedCount = insertedVertId.size();
            for(int i=0; i <insertedCount; ++i)
            {
                Vertex& vertex = vert[insertedVertId[i]];
                TetListNode* node = vertex.tetList.head;
                while(node != nullptr)
                {
                    Tetrahedron* tet = node->tet;
                    if(tet->visitTime < _currentVisitTime)
                    {
                        tet->visitTime = _currentVisitTime;
                        if(intersects(v, tet))
                        {
                            return tet;
                        }
                    }

                    node = node->next;
                }
            }


            // Push neighbors
            switch (dir)
            {
            case STATIC:
                if(cId.z != 0)
                    _baseQueue.push_back(make_pair(cId, DOWN));

                _baseQueue.push_back(make_pair(c, BACK));
                _baseQueue.push_back(make_pair(c, RIGHT));
                _baseQueue.push_back(make_pair(c, FRONT));
                _baseQueue.push_back(make_pair(c, LEFT));
                _baseQueue.push_back(make_pair(c, BACK_RIGHT));
                _baseQueue.push_back(make_pair(c, FRONT_RIGHT));
                _baseQueue.push_back(make_pair(c, FRONT_LEFT));
                _baseQueue.push_back(make_pair(c, BACK_LEFT));

                if(cId.z != 0)
                {
                    _baseQueue.push_back(make_pair(c, BACK_DOWN));
                    _baseQueue.push_back(make_pair(c, RIGHT_DOWN));
                    _baseQueue.push_back(make_pair(c, FRONT_DOWN));
                    _baseQueue.push_back(make_pair(c, LEFT_DOWN));
                    _baseQueue.push_back(make_pair(c, BACK_RIGHT_DOWN));
                    _baseQueue.push_back(make_pair(c, FRONT_RIGHT_DOWN));
                    _baseQueue.push_back(make_pair(c, FRONT_LEFT_DOWN));
                    _baseQueue.push_back(make_pair(c, BACK_LEFT_DOWN));
                }
                break;

            case BACK :
                _baseQueue.push_back(make_pair(c, BACK));
                break;

            case BACK_RIGHT :
                _baseQueue.push_back(make_pair(c, BACK));
                _baseQueue.push_back(make_pair(c, RIGHT));
                _baseQueue.push_back(make_pair(c, BACK_RIGHT));
                break;

            case RIGHT :
                _baseQueue.push_back(make_pair(c, RIGHT));
                break;

            case FRONT_RIGHT :
                _baseQueue.push_back(make_pair(c, RIGHT));
                _baseQueue.push_back(make_pair(c, FRONT));
                _baseQueue.push_back(make_pair(c, FRONT_RIGHT));
                break;

            case FRONT :
                _baseQueue.push_back(make_pair(c, FRONT));
                break;

            case FRONT_LEFT :
                _baseQueue.push_back(make_pair(c, FRONT));
                _baseQueue.push_back(make_pair(c, LEFT));
                _baseQueue.push_back(make_pair(c, FRONT_LEFT));
                break;

            case LEFT :
                _baseQueue.push_back(make_pair(c, LEFT));
                break;

            case BACK_LEFT :
                _baseQueue.push_back(make_pair(c, BACK));
                _baseQueue.push_back(make_pair(c, LEFT));
                _baseQueue.push_back(make_pair(c, BACK_LEFT));
                break;


            case DOWN:
                if(c.z != 0)
                {
                    _baseQueue.push_back(make_pair(c, DOWN));
                }
                break;

            case BACK_DOWN :
                _baseQueue.push_back(make_pair(c, BACK));
                if(c.z != 0)
                {
                    _baseQueue.push_back(make_pair(c, DOWN));
                    _baseQueue.push_back(make_pair(c, BACK_DOWN));
                }
                break;

            case BACK_RIGHT_DOWN :
                _baseQueue.push_back(make_pair(c, BACK));
                _baseQueue.push_back(make_pair(c, RIGHT));
                _baseQueue.push_back(make_pair(c, BACK_RIGHT));
                if(c.z != 0)
                {
                    _baseQueue.push_back(make_pair(c, DOWN));
                    _baseQueue.push_back(make_pair(c, BACK_DOWN));
                    _baseQueue.push_back(make_pair(c, RIGHT_DOWN));
                    _baseQueue.push_back(make_pair(c, BACK_RIGHT_DOWN));
                }

                break;

            case RIGHT_DOWN :
                _baseQueue.push_back(make_pair(c, RIGHT));
                if(c.z != 0)
                {
                    _baseQueue.push_back(make_pair(c, DOWN));
                    _baseQueue.push_back(make_pair(c, RIGHT_DOWN));
                }
                break;

            case FRONT_RIGHT_DOWN :
                _baseQueue.push_back(make_pair(c, FRONT));
                _baseQueue.push_back(make_pair(c, RIGHT));
                _baseQueue.push_back(make_pair(c, FRONT_RIGHT));
                if(c.z != 0)
                {
                    _baseQueue.push_back(make_pair(c, DOWN));
                    _baseQueue.push_back(make_pair(c, FRONT_DOWN));
                    _baseQueue.push_back(make_pair(c, RIGHT_DOWN));
                    _baseQueue.push_back(make_pair(c, FRONT_RIGHT_DOWN));
                }
                break;

            case FRONT_DOWN :
                _baseQueue.push_back(make_pair(c, FRONT));
                if(c.z != 0)
                {
                    _baseQueue.push_back(make_pair(c, DOWN));
                    _baseQueue.push_back(make_pair(c, FRONT_DOWN));
                }
                break;

            case FRONT_LEFT_DOWN :
                _baseQueue.push_back(make_pair(c, FRONT));
                _baseQueue.push_back(make_pair(c, LEFT));
                _baseQueue.push_back(make_pair(c, FRONT_LEFT));
                if(c.z != 0)
                {
                    _baseQueue.push_back(make_pair(c, DOWN));
                    _baseQueue.push_back(make_pair(c, FRONT_DOWN));
                    _baseQueue.push_back(make_pair(c, LEFT_DOWN));
                    _baseQueue.push_back(make_pair(c, FRONT_LEFT_DOWN));
                }
                break;

            case LEFT_DOWN :
                _baseQueue.push_back(make_pair(c, LEFT));
                if(c.z != 0)
                {
                    _baseQueue.push_back(make_pair(c, DOWN));
                    _baseQueue.push_back(make_pair(c, LEFT_DOWN));
                }
                break;

            case BACK_LEFT_DOWN :
                _baseQueue.push_back(make_pair(c, BACK));
                _baseQueue.push_back(make_pair(c, LEFT));
                _baseQueue.push_back(make_pair(c, BACK_LEFT));
                if(c.z != 0)
                {
                    _baseQueue.push_back(make_pair(c, DOWN));
                    _baseQueue.push_back(make_pair(c, BACK_DOWN));
                    _baseQueue.push_back(make_pair(c, LEFT_DOWN));
                    _baseQueue.push_back(make_pair(c, BACK_LEFT_DOWN));
                }
                break;
            }
        }
    }

    bool isBaseTetrahedronFound = false;
    assert(isBaseTetrahedronFound);
}

void Mesh::findDelaunayBall(int vId, Tetrahedron* base, std::unordered_set<Triangle>& ball)
{
    const glm::dvec3& v = vert[vId].p;
    Vertex& v0 = vert[base->v[0]];
    Vertex& v1 = vert[base->v[1]];
    Vertex& v2 = vert[base->v[2]];
    Vertex& v3 = vert[base->v[3]];

    _ballQueue.clear();
    _ballQueue.push_back(&v0);
    _ballQueue.push_back(&v1);
    _ballQueue.push_back(&v2);
    _ballQueue.push_back(&v3);

    ++_currentVisitTime;
    v0.visitTime = _currentVisitTime;
    v1.visitTime = _currentVisitTime;
    v2.visitTime = _currentVisitTime;
    v3.visitTime = _currentVisitTime;

    ball.insert(base->t0());
    ball.insert(base->t1());
    ball.insert(base->t2());
    ball.insert(base->t3());
    removeTetrahedronGrid(base);

    int hit = 0;
    int miss = 0;

    for(int qId = 0; qId < _ballQueue.size(); ++qId)
    {
        Vertex* vertex = _ballQueue[qId];

        TetListNode* node = vertex->tetList.head;
        while(node != nullptr)
        {
            Tetrahedron* tet = node->tet;
            node = node->next;

            if(tet->visitTime < _currentVisitTime)
            {
                tet->visitTime = _currentVisitTime;

                if(intersects(v, tet))
                {
                    ++hit;

                    decltype(ball.insert(base->t0())) it;

                    it = ball.insert(tet->t0());
                    if(!it.second)
                        ball.erase(it.first);

                    it = ball.insert(tet->t1());
                    if(!it.second)
                        ball.erase(it.first);

                    it = ball.insert(tet->t2());
                    if(!it.second)
                        ball.erase(it.first);

                    it = ball.insert(tet->t3());
                    if(!it.second)
                        ball.erase(it.first);


                    Vertex* tv0 = &vert[tet->v[0]];
                    if(tv0->visitTime < _currentVisitTime)
                    {
                        tv0->visitTime = _currentVisitTime;
                        _ballQueue.push_back(tv0);
                    }

                    Vertex* tv1 = &vert[tet->v[1]];
                    if(tv1->visitTime < _currentVisitTime)
                    {
                        tv1->visitTime = _currentVisitTime;
                        _ballQueue.push_back(tv1);
                    }

                    Vertex* tv2 = &vert[tet->v[2]];
                    if(tv2->visitTime < _currentVisitTime)
                    {
                        tv2->visitTime = _currentVisitTime;
                        _ballQueue.push_back(tv2);
                    }

                    Vertex* tv3 = &vert[tet->v[3]];
                    if(tv3->visitTime < _currentVisitTime)
                    {
                        tv3->visitTime = _currentVisitTime;
                        _ballQueue.push_back(tv3);
                    }


                    removeTetrahedronGrid(tet);
                }
                else
                {
                    ++miss;
                }
            }
        }
    }

    cout << "(" << hit << ", " << miss << ")\t";
}

void Mesh::remeshDelaunayBall(const glm::ivec3& cId, int vId, const std::unordered_set<Triangle>& ball)
{
    for(auto t : ball)
    {
        insertTetrahedronGrid(vId, t.v[0], t.v[1], t.v[2]);
    }
}

void Mesh::insertTetrahedronGrid(int v0, int v1, int v2, int v3)
{
    Tetrahedron* tet = _acquireTetrahedron(v0, v1, v2, v3);
    tet->visitTime = _currentVisitTime;

    // Literally insert in the grid and mesh
    vert[tet->v[0]].tetList.addTet(tet);
    vert[tet->v[1]].tetList.addTet(tet);
    vert[tet->v[2]].tetList.addTet(tet);
    vert[tet->v[3]].tetList.addTet(tet);

    // Compute tetrahedron circumcircle
    const glm::dvec3& A = vert[tet->v[0]].p;
    const glm::dvec3& B = vert[tet->v[1]].p;
    const glm::dvec3& C = vert[tet->v[2]].p;
    const glm::dvec3& D = vert[tet->v[3]].p;
    glm::dmat3 S = glm::transpose(
        glm::dmat3(A-B, B-C, C-D));
    glm::dvec3 R(
        (glm::dot(A, A) - glm::dot(B, B)) / 2.0f,
        (glm::dot(B, B) - glm::dot(C, C)) / 2.0f,
        (glm::dot(C, C) - glm::dot(D, D)) / 2.0f);

    double SdetInv = 1.0 / glm::determinant(S);
    double Sx = glm::determinant(glm::dmat3(R, S[1], S[2]));
    double Sy = glm::determinant(glm::dmat3(S[0], R, S[2]));
    double Sz = glm::determinant(glm::dmat3(S[0], S[1], R));

    tet->circumCenter = glm::dvec3(Sx, Sy, Sz) * SdetInv;
    glm::dvec3 dist = A - tet->circumCenter;
    tet->circumRadius2 = glm::dot(dist, dist);
}

void Mesh::removeTetrahedronGrid(Tetrahedron* tet)
{
    vert[tet->v[0]].tetList.delTet(tet);
    vert[tet->v[1]].tetList.delTet(tet);
    vert[tet->v[2]].tetList.delTet(tet);
    vert[tet->v[3]].tetList.delTet(tet);
    _disposeTetrahedron(tet);
}

void Mesh::makeTetrahedronPositive(Tetrahedron* tet)
{
    const glm::dvec3& A = vert[tet->v[0]].p;
    const glm::dvec3& B = vert[tet->v[1]].p;
    const glm::dvec3& C = vert[tet->v[2]].p;
    const glm::dvec3& D = vert[tet->v[3]].p;

    glm::dvec3 d10(A - B);
    glm::dvec3 d12(C - B);
    glm::dvec3 d23(D - C);
    glm::dmat3 triple(d10, d12, d23);
    if(glm::determinant(triple) < 0)
    {
        std::swap(tet->v[2], tet->v[3]);
    }
}

void Mesh::tearDownGrid()
{
    ++_currentVisitTime;

    // Clear grid
    grid.clear();

    // Collect tetrahedrons
    int vertCount = vert.size();
    for(int v=0; v < vertCount; ++v)
    {
        Vertex& vertex = vert[v];
        TetListNode* node = vertex.tetList.head;
        while(node != nullptr)
        {
            Tetrahedron* tet = node->tet;
            if(tet->visitTime < _currentVisitTime)
            {
                tet->visitTime = _currentVisitTime;
                makeTetrahedronPositive(tet);
                tetra.push_back(tet);
            }

            node = node->next;
        }

        vertex.tetList.clrTet();
    }

    TetList::releaseMemoryPool();
    _releaseTetrahedronMemoryPool();

    cout << "Tetrahedrons / Vertex = " <<
            tetra.size() / (double) vert.size() << endl;
}

void Mesh::compileAdjacencyLists()
{
    neighbors.clear();
    neighbors.resize(vertCount());

    for(auto& v : vert)
    {
        v.isBoundary = false;
    }

    for(const auto& tet : tetra)
    {
        if(isExternalTetraHedron(tet))
        {
            vert[tet->v[0]].isBoundary = true;
            vert[tet->v[1]].isBoundary = true;
            vert[tet->v[2]].isBoundary = true;
            vert[tet->v[3]].isBoundary = true;
            continue;
        }

        int verts[][2] = {
            {tet->v[0], tet->v[1]},
            {tet->v[0], tet->v[2]},
            {tet->v[0], tet->v[3]},
            {tet->v[1], tet->v[2]},
            {tet->v[1], tet->v[3]},
            {tet->v[2], tet->v[3]},
        };

        for(int e=0; e<6; ++e)
        {
            bool isPresent = false;
            int firstVert = verts[e][0];
            int secondVert = verts[e][1];
            int neighborCount = neighbors[firstVert].size();
            for(int n=0; n < neighborCount; ++n)
            {
                if(secondVert == neighbors[firstVert][n])
                {
                    isPresent = true;
                    break;
                }
            }

            if(!isPresent)
            {
                neighbors[firstVert].push_back(secondVert);
                neighbors[secondVert].push_back(firstVert);
            }
        }
    }
}
