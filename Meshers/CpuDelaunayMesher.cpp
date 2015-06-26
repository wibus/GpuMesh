#include "CpuDelaunayMesher.h"

#include <chrono>
#include <iostream>
#include <algorithm>

#include <GLM/glm.hpp>
#include <GLM/gtc/random.hpp>

#include "DataStructures/TetList.h"

using namespace std;


struct Vertex
{
    Vertex() {}
    Vertex(const glm::dvec3& pos) :
        p(pos)
    {}

    glm::dvec3 p;
    TetList tetList;

    // Algo flag
    int visitTime;
};


struct GridCell
{
    std::vector<int> insertedVertId;
    std::vector<int> waitingVertId;
};

const glm::ivec3 DIR[DIR_COUNT] = {
    glm::ivec3( 0,  0,  0),
    glm::ivec3(-1,  0,  0), glm::ivec3(-1, -1,  0),
    glm::ivec3( 0, -1,  0), glm::ivec3( 1, -1,  0),
    glm::ivec3( 1,  0,  0), glm::ivec3( 1,  1,  0),
    glm::ivec3( 0,  1,  0), glm::ivec3(-1,  1,  0),

    glm::ivec3( 0,  0, -1),
    glm::ivec3(-1,  0, -1), glm::ivec3(-1, -1, -1),
    glm::ivec3( 0, -1, -1), glm::ivec3( 1, -1, -1),
    glm::ivec3( 1,  0, -1), glm::ivec3( 1,  1, -1),
    glm::ivec3( 0,  1, -1), glm::ivec3(-1,  1, -1)
};


CpuDelaunayMesher::CpuDelaunayMesher(Mesh& mesh, unsigned int vertCount) :
    AbstractMesher(mesh, vertCount)
{

}

CpuDelaunayMesher::~CpuDelaunayMesher()
{

}

void CpuDelaunayMesher::triangulateDomain()
{
    chrono::high_resolution_clock::time_point startTime, endTime;
    startTime = chrono::high_resolution_clock::now();
    insertVertices();
    endTime = chrono::high_resolution_clock::now();

    chrono::microseconds dt;
    dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Total meshing time = " << dt.count() / 1000.0 << "ms" << endl;
}

void CpuDelaunayMesher::genBoundingMesh()
{
    const double a = 20.0;

    std::vector<glm::dvec3> vertices;
    vertices.push_back(glm::dvec3(-a, -a,  a));
    vertices.push_back(glm::dvec3( a, -a,  a));
    vertices.push_back(glm::dvec3(-a,  a,  a));
    vertices.push_back(glm::dvec3( a,  a,  a));
    vertices.push_back(glm::dvec3(-a, -a, -a));
    vertices.push_back(glm::dvec3( a, -a, -a));
    vertices.push_back(glm::dvec3(-a,  a, -a));
    vertices.push_back(glm::dvec3( a,  a, -a));

    std::vector<Tetrahedron> tetrahedron;
    tetrahedron.push_back(Tetrahedron(0, 1, 2, 4));
    tetrahedron.push_back(Tetrahedron(5, 4, 7, 1));
    tetrahedron.push_back(Tetrahedron(3, 1, 7, 2));
    tetrahedron.push_back(Tetrahedron(6, 2, 7, 4));
    tetrahedron.push_back(Tetrahedron(4, 1, 2, 7));


    int vertCount = vertices.size();
    vert.resize(vertCount);
    for(int i=0; i<vertCount; ++i)
    {
        vert[i] = Vertex(vertices[i]);
    }

    int tetCount = tetrahedron.size();
    tetra.resize(tetCount);
    for(int i=0; i<tetCount; ++i)
    {
        const Tetrahedron& tet = tetrahedron[i];
        tetra[i] = _tetPool.acquireTetrahedron(
            tet.v[0], tet.v[1], tet.v[2], tet.v[3]);
    }

    _externalVertCount = vertCount;
}

void CpuDelaunayMesher::genVertices(std::vector<glm::dvec3>& vertices)
{
    double sphereRadius = 1.0;
    glm::dvec3 cMin(-sphereRadius);
    glm::dvec3 cMax( sphereRadius);


    //* Box distribution
    vertices.resize(_vertCount);
    int surfCount = glm::sqrt(_vertCount) * 10;
    int padding = glm::pow(_vertCount, 1/3.0);
    for(int iv=0; iv < surfCount; ++iv)
    {
        glm::dvec3 val = glm::linearRand(cMin, cMax);
        val[iv%3] = glm::mix(-1.0, 1.0, (double)(iv%2));
        vertices[iv] = val;
    }

    for(int iv=surfCount; iv<_vertCount; ++iv)
        vertices[iv] = glm::linearRand(cMin, cMax) * (1.0 - 1.0 / padding);

    //*/

    /* Sphere distribution
    vertices.resize(_vertCount);
    for(int iv=0; iv<_vertCount; ++iv)
        vertices[iv] = glm::ballRand(sphereRadius * 1.41);
    //*/


    /* Paraboloid cap
    int stackCount = glm::pow(_vertCount/2.3, 1/3.0);
    glm::dvec3 minPerturb(-0.1 / stackCount);
    glm::dvec3 maxPerturb(0.1 / stackCount);
    for(int s=0; s <= stackCount; ++s)
    {
        double sProg = s / (double) stackCount;
        double height = sProg;
        vertices.push_back(glm::dvec3(0, 0, height));

        for(int r=1; r <= s; ++r)
        {
            double rProg = r / (double) s;
            double ringRadius = r / (double) stackCount;
            double ringHeight = height * (1.0 - rProg*rProg);
            double ringVertCount = stackCount * glm::sqrt(1.0 + r);
            for(int v=0; v < ringVertCount; ++v)
            {
                double vProg = v / (double) ringVertCount;
                double vAngle = glm::pi<double>() * vProg * 2;
                vertices.push_back(glm::dvec3(
                    glm::cos(vAngle) * ringRadius,
                    glm::sin(vAngle) * ringRadius * 0.75,
                    ringHeight)
                                   +
                    glm::linearRand(minPerturb, maxPerturb)
                     * (1.1 - rProg) * (1.1 - sProg));
            }
        }
    }
    //*/
}

void CpuDelaunayMesher::insertVertices()
{
    genBoundingMesh();

    std::vector<glm::dvec3> vertices;
    genVertices(vertices);

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

    vertices.clear();
    vertices.shrink_to_fit();

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

void CpuDelaunayMesher::initializeGrid(int idStart, int idEnd)
{
    // Compute dimensions
    const double VERT_PER_CELL = 5.0;
    int vCount = idEnd - idStart;
    int sideLen = (int) glm::ceil(glm::pow(vCount / VERT_PER_CELL, 1/3.0));
    int height = (int) glm::ceil(vCount / (sideLen * sideLen * VERT_PER_CELL) );
    gridSize = glm::ivec3(sideLen, sideLen, height);
    int cellCount = gridSize.x*gridSize.y*gridSize.z;
    std::cout << "Grid size: "
              << gridSize.x << "x"
              << gridSize.y << "x"
              << gridSize.z << endl;
    std::cout << "Cell density: vert count/cell count = " <<
                 (idEnd-idStart) << " / " << cellCount << " = " <<
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
    int tetCount = tetra.size();
    for(int i=0; i < tetCount; ++i)
    {
        Tetrahedron* tet = tetra[i];
        insertTetrahedronGrid(
            tet->v[0],
            tet->v[1],
            tet->v[2],
            tet->v[3]);

        _tetPool.disposeTetrahedron(tet);
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

    _ball.reset(307);
}


void CpuDelaunayMesher::insertCell(const glm::ivec3& cId)
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

void CpuDelaunayMesher::insertVertexGrid(const glm::ivec3& cId, int vId)
{
    _ball.clear();
    findDelaunayBall(cId, vId);
    remeshDelaunayBall(vId);
}

Tetrahedron* CpuDelaunayMesher::findBaseTetrahedron(const glm::ivec3& cId, int vId)
{
    const glm::dvec3& v = vert[vId].p;

    _baseQueue.clear();
    ++_currentVisitTime;

    _baseQueue.push_back(make_pair(cId, STATIC));
    for(int qId = 0; qId < _baseQueue.size(); ++qId)
    {
        const EDir& dir = _baseQueue[qId].second;
        glm::ivec3 c = _baseQueue[qId].first + DIR[dir];

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

void CpuDelaunayMesher::findDelaunayBall(const glm::ivec3& cId, int vId)
{
    Tetrahedron* base = findBaseTetrahedron(cId, vId);
    Vertex& v0 = vert[base->v[0]];
    Vertex& v1 = vert[base->v[1]];
    Vertex& v2 = vert[base->v[2]];
    Vertex& v3 = vert[base->v[3]];

    _ballQueue.clear();
    _ballQueue.push_back(&v0);
    _ballQueue.push_back(&v1);
    _ballQueue.push_back(&v2);
    _ballQueue.push_back(&v3);

    _ball.xOrTri(base->t0());
    _ball.xOrTri(base->t1());
    _ball.xOrTri(base->t2());
    _ball.xOrTri(base->t3());

    removeTetrahedronGrid(base);
    const glm::dvec3& v = vert[vId].p;

    ++_currentVisitTime;
    v0.visitTime = _currentVisitTime;
    v1.visitTime = _currentVisitTime;
    v2.visitTime = _currentVisitTime;
    v3.visitTime = _currentVisitTime;


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
                    _ball.xOrTri(tet->t0());
                    _ball.xOrTri(tet->t1());
                    _ball.xOrTri(tet->t2());
                    _ball.xOrTri(tet->t3());


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
            }
        }
    }
}

void CpuDelaunayMesher::remeshDelaunayBall(int vId)
{
    const std::vector<Triangle>& tris = _ball.gather();
    int triCount = tris.size();
    for(int i=0; i < triCount; ++i)
    {
        const Triangle& t = tris[i];
        insertTetrahedronGrid(vId, t.v[0], t.v[1], t.v[2]);
    }
}

void CpuDelaunayMesher::insertTetrahedronGrid(int v0, int v1, int v2, int v3)
{
    Tetrahedron* tet = _tetPool.acquireTetrahedron(v0, v1, v2, v3);
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

void CpuDelaunayMesher::removeTetrahedronGrid(Tetrahedron* tet)
{
    vert[tet->v[0]].tetList.delTet(tet);
    vert[tet->v[1]].tetList.delTet(tet);
    vert[tet->v[2]].tetList.delTet(tet);
    vert[tet->v[3]].tetList.delTet(tet);
    _tetPool.disposeTetrahedron(tet);
}

void CpuDelaunayMesher::tearDownGrid()
{
    ++_currentVisitTime;

    // Clear grid
    grid.clear();


    // Release memory pools
    _ball.releaseMemoryPool();
    _tetPool.releaseMemoryPool();
    TetList::releaseMemoryPool();


    // Copy vertices in mesh
    int delaunayVertCount = vert.size();
    int meshVertCount = delaunayVertCount - _externalVertCount;

    // Shorthands
    decltype(_mesh.vert)& meshVert = _mesh.vert;
    decltype(_mesh.tetra)& meshTetra = _mesh.tetra;
    decltype(_mesh.topo)& meshTopo = _mesh.topo;

    meshTetra.clear();
    meshVert.resize(meshVertCount);
    meshTopo.resize(meshVertCount);
    for(int i = _externalVertCount; i < delaunayVertCount; ++i)
    {
        Vertex& dVert = vert[i];
        MeshVert& mVert = meshVert[i-_externalVertCount];

        mVert.p = dVert.p;

        TetListNode* node = dVert.tetList.head;
        while(node != nullptr)
        {
            Tetrahedron* tet = node->tet;
            if(tet->visitTime < _currentVisitTime)
            {
                tet->visitTime = _currentVisitTime;
                makeTetrahedronPositive(tet);

                MeshTet meshTet(
                    tet->v[0] - _externalVertCount,
                    tet->v[1] - _externalVertCount,
                    tet->v[2] - _externalVertCount,
                    tet->v[3] - _externalVertCount);

                if(meshTet[0] < 0 || meshTet[1] < 0 ||
                   meshTet[2] < 0 || meshTet[3] < 0)
                {
                    // It's a bounding tetrahedron
                    if(meshTet[0] >= 0) meshTopo[meshTet[0]].isFixed = true;
                    if(meshTet[1] >= 0) meshTopo[meshTet[1]].isFixed = true;
                    if(meshTet[2] >= 0) meshTopo[meshTet[2]].isFixed = true;
                    if(meshTet[3] >= 0) meshTopo[meshTet[3]].isFixed = true;
                }
                else
                {
                    // It's a real tetrahedron
                    meshTetra.push_back(meshTet);
                }

                // Tetrahedrons are not directly deleted
                // They must remain allocated to access their _visitTime member
                // Last call to _tetPool.releaseMemoryPool() actually delete them
                _tetPool.disposeTetrahedron(tet);
            }

            node = node->next;
        }

        dVert.tetList.clrTet();
    }
    meshTetra.shrink_to_fit();


    // Discard unused memory
    vert.clear();
    vert.shrink_to_fit();

    tetra.clear();
    tetra.shrink_to_fit();


    // Release just deleted tetrahedrons
    _tetPool.releaseMemoryPool();


    cout << "Elements / Vertices = " <<
            _mesh.elemCount() << " / " << _mesh.vertCount() << " = " <<
            _mesh.elemCount()  / (double) _mesh.vertCount() << endl;
}

bool CpuDelaunayMesher::intersects(const glm::dvec3& v, Tetrahedron* tet)
{
    glm::dvec3 dist = v - tet->circumCenter;
    if(glm::dot(dist, dist) < tet->circumRadius2)
        return true;

    return false;
}

void CpuDelaunayMesher::makeTetrahedronPositive(Tetrahedron* tet)
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
