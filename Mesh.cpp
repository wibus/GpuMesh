#include "Mesh.h"

#include <map>
#include <iostream>

using namespace std;


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

        glm::dvec3 norms[] = {
            glm::normalize(glm::cross(verts[1] - verts[0], verts[2] - verts[1])),
            glm::normalize(glm::cross(verts[2] - verts[0], verts[3] - verts[2])),
            glm::normalize(glm::cross(verts[3] - verts[0], verts[1] - verts[3])),
            glm::normalize(glm::cross(verts[3] - verts[1], verts[2] - verts[3])),
        };

        double quality = tetrahedronQuality(tet);

        qualityMean = (qualityMean * qualityCount + quality) / (qualityCount + 1);
        double qualityMeanDist = qualityMean - quality;
        qualityVar = (qualityVar * qualityCount + qualityMeanDist*qualityMeanDist) / (qualityCount + 1);
        ++qualityCount;

        pushTriangle(vertices, normals, triEdges, tetQualities,
                     verts[0], verts[1], verts[2], norms[0], quality);
        pushTriangle(vertices, normals, triEdges, tetQualities,
                     verts[0], verts[2], verts[3], norms[1], quality);
        pushTriangle(vertices, normals, triEdges, tetQualities,
                     verts[0], verts[3], verts[1], norms[2], quality);
        pushTriangle(vertices, normals, triEdges, tetQualities,
                     verts[1], verts[3], verts[2], norms[3], quality);
    }
}


void Mesh::compileAdjacencyLists(
        std::vector<std::vector<int>>& neighbors)
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

    glm::dvec3 dim = cMax- cMin;
    cMin -= dim / 1000000.0;
    cMax += dim / 1000000.0;

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
    cDim = (cMax - cMin);
    glm::dvec3 floatSize = glm::dvec3(gridSize);
    for(int vId=idStart; vId<idEnd; ++vId)
    {
        const glm::dvec3& v = vert[vId].p;
        glm::ivec3 bin = glm::ivec3((v - cMin) / (cDim) * floatSize);
        grid[bin.z][bin.y][bin.x].vertId.push_back(vId);
    }

    // Put starting tetrahedrons in the first cell
    const glm::ivec3 FIRST_CELL(0, 0, 0);
    for(auto tet : tetra)
    {
        insertTetrahedronGrid(FIRST_CELL,
            tet->v[0], tet->v[1], tet->v[2], tet->v[3]);
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
                glm::dvec3 cellCorner = floatBin * cDim + cMin;

                std::sort(grid[k][j][i].vertId.begin(),
                          grid[k][j][i].vertId.end(),
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
    pullupTetrahedrons(cId);

    GridCell& cell = grid[cId.z][cId.y][cId.x];
    int vertCount = cell.vertId.size();
    for(int vId=0; vId<vertCount; ++vId)
    {
        insertVertexGrid(cId, cell.vertId[vId]);
    }
}

void Mesh::pullupTetrahedrons(const glm::ivec3& cId)
{
    GridCell& cell = grid[cId.z][cId.y][cId.x];

    if(cId.z > 0)
    {
        std::unordered_set<Tetrahedron*>& cellTets = cell.tetra;

        GridCell& floor = grid[cId.z-1][cId.y][cId.x];
        std::unordered_set<Tetrahedron*>& floorTets = floor.tetra;
        auto tetIt = floorTets.begin();

        double zThreshold = (cId.z / (double) gridSize.z) * cDim.z + cMin.z;

        while(tetIt != floorTets.end())
        {
            Tetrahedron* tet = *tetIt;

            if(vert[tet->v[0]].p.z > zThreshold)
            {
                tet->cId = cId;
                cellTets.insert(tet);
                tetIt = floorTets.erase(tetIt);
            }
            else if(vert[tet->v[1]].p.z > zThreshold)
            {
                tet->cId = cId;
                cellTets.insert(tet);
                tetIt = floorTets.erase(tetIt);
            }
            else if(vert[tet->v[2]].p.z > zThreshold)
            {
                tet->cId = cId;
                cellTets.insert(tet);
                tetIt = floorTets.erase(tetIt);
            }
            else if(vert[tet->v[3]].p.z > zThreshold)
            {
                tet->cId = cId;
                cellTets.insert(tet);
                tetIt = floorTets.erase(tetIt);
            }
            else
            {
                ++tetIt;
            }
        }
    }
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
    const glm::ivec3 BACK(-1, 0, 0);
    const glm::ivec3 BACK_RIGHT(-1, -1, 0);
    const glm::ivec3 RIGHT(0, -1, 0);
    const glm::ivec3 FRONT_RIGHT(1, -1, 0);

    const glm::ivec3 FRONT(1, 0, 0);
    const glm::ivec3 FRONT_LEFT(1, 1, 0);
    const glm::ivec3 LEFT(0, 1, 0);
    const glm::ivec3 BACK_LEFT(-1, 1, 0);

    const glm::ivec3 FRONT_DOWN(1, 0, -1);
    const glm::ivec3 FRONT_LEFT_DOWN(1, 1, -1);
    const glm::ivec3 LEFT_DOWN(0, 1, -1);
    const glm::ivec3 BACK_LEFT_DOWN(-1, 1, -1);

    _baseQueue.clear();
    _baseQueue.push_back(make_pair(cId,               EDir::STATIC));
    _baseQueue.push_back(make_pair(cId + BACK,        EDir::BACK));
    _baseQueue.push_back(make_pair(cId + BACK_RIGHT,  EDir::BACK_RIGHT));
    _baseQueue.push_back(make_pair(cId + RIGHT,       EDir::RIGHT));
    _baseQueue.push_back(make_pair(cId + FRONT_RIGHT, EDir::FRONT_RIGHT));
    if(cId.z > 0)
    {
        _baseQueue.push_back(make_pair(cId + FRONT_DOWN,      EDir::FRONT));
        _baseQueue.push_back(make_pair(cId + FRONT_LEFT_DOWN, EDir::FRONT_LEFT));
        _baseQueue.push_back(make_pair(cId + LEFT_DOWN,       EDir::LEFT));
        _baseQueue.push_back(make_pair(cId + BACK_LEFT_DOWN,  EDir::BACK_LEFT));
    }


    int qId = 0;
    while(qId != _baseQueue.size())
    {
        const glm::ivec3& c = _baseQueue[qId].first;

        if(0 <= c.x && c.x < gridSize.x &&
           0 <= c.y && c.y < gridSize.y)
        {
            GridCell& cell = grid[c.z][c.y][c.x];
            for(auto tet : cell.tetra)
            {
                if(isBase(vId, tet))
                {
                    return tet;
                }
            }

            // Push neighbors
            switch (_baseQueue[qId].second)
            {
            case EDir::BACK :
                _baseQueue.push_back(make_pair(c + BACK, EDir::BACK));
                break;

            case EDir::BACK_RIGHT :
                _baseQueue.push_back(make_pair(c + BACK, EDir::BACK));
                _baseQueue.push_back(make_pair(c + BACK_RIGHT, EDir::BACK_RIGHT));
                _baseQueue.push_back(make_pair(c + RIGHT, EDir::RIGHT));
                break;

            case EDir::RIGHT :
                _baseQueue.push_back(make_pair(c + RIGHT, EDir::RIGHT));
                break;

            case EDir::FRONT_RIGHT :
                _baseQueue.push_back(make_pair(c + RIGHT, EDir::RIGHT));
                _baseQueue.push_back(make_pair(c + FRONT_RIGHT, EDir::FRONT_RIGHT));
                _baseQueue.push_back(make_pair(c + FRONT, EDir::FRONT));
                break;

            case EDir::FRONT :
                _baseQueue.push_back(make_pair(c + FRONT, EDir::FRONT));
                break;

            case EDir::FRONT_LEFT :
                _baseQueue.push_back(make_pair(c + FRONT, EDir::FRONT));
                _baseQueue.push_back(make_pair(c + FRONT_LEFT, EDir::FRONT_LEFT));
                _baseQueue.push_back(make_pair(c + LEFT, EDir::LEFT));
                break;

            case EDir::LEFT :
                _baseQueue.push_back(make_pair(c + LEFT, EDir::LEFT));
                break;

            case EDir::BACK_LEFT :
                _baseQueue.push_back(make_pair(c + LEFT, EDir::LEFT));
                _baseQueue.push_back(make_pair(c + BACK_LEFT, EDir::BACK_LEFT));
                _baseQueue.push_back(make_pair(c + BACK, EDir::BACK));
                break;
            }
        }

        ++qId;
    }

    bool isBaseTetrahedronFound = false;
    assert(isBaseTetrahedronFound);
}

bool Mesh::isBase(int vId, Tetrahedron* tet)
{
    glm::dmat4 D(
        glm::dvec4(vert[tet->v[0]].p, 1),
        glm::dvec4(vert[tet->v[1]].p, 1),
        glm::dvec4(vert[tet->v[2]].p, 1),
        glm::dvec4(vert[tet->v[3]].p, 1));

    glm::dvec4 v4(vert[vId].p, 1);
    glm::dmat4 Dv(v4, D[1], D[2], D[3]);

    if(glm::determinant(Dv) < 0)
        return false;
    Dv[0] = D[0];

    Dv[1] = v4;
    if(glm::determinant(Dv) < 0)
        return false;
    Dv[1] = D[1];

    Dv[2] = v4;
    if(glm::determinant(Dv) < 0)
        return false;
    Dv[2] = D[2];

    Dv[3] = v4;
    if(glm::determinant(Dv) < 0)
        return false;

    return true;
}


void Mesh::findDelaunayBall(int vId, Tetrahedron* base, std::unordered_set<Triangle>& ball)
{
    glm::dvec3 v = vert[vId].p;

    _ballPreserved.clear();
    _ballTouched.clear();

    _ballQueue.clear();
    _ballQueue.push_back(base);
    base->flag = true;

    int qId = 0;
    while(qId < _ballQueue.size())
    {
        Tetrahedron* tet = _ballQueue[qId];

        glm::dvec3 dist = v - tet->circumCenter;
        if(glm::dot(dist, dist) < tet->circumRadius2)
        {
            Triangle tri[] {tet->t0(), tet->t1(), tet->t2(), tet->t3()};

            for(int i=0; i<4; ++i)
            {
                auto it = ball.insert(tri[i]);
                if(!it.second)
                {
                    ball.erase(it.first);
                }

                Vertex& triVert = vert[tet->v[i]];
                if(!triVert.flag)
                {
                    triVert.flag = true;
                    _ballTouched.push_back(&triVert);

                    for(auto neighbor : triVert.tetra)
                    {
                        if(!neighbor->flag)
                        {
                            neighbor->flag = true;
                            _ballQueue.push_back(neighbor);
                        }
                    }
                }
            }

            removeTetrahedronGrid(tet);
        }
        else
        {
            _ballPreserved.push_back(tet);
        }

        ++qId;
    }


    // Reset algo flag on preserved tetrahedrons
    int preservedCount = _ballPreserved.size();
    for(int i=0; i<preservedCount; ++i)
    {
        _ballPreserved[i]->flag = false;
    }

    int touchedCount = _ballTouched.size();
    for(int i=0; i<touchedCount; ++i)
    {
        _ballTouched[i]->flag = false;
    }
}

void Mesh::remeshDelaunayBall(const glm::ivec3& cId, int vId, const std::unordered_set<Triangle>& ball)
{
    for(auto t : ball)
    {
        insertTetrahedronGrid(cId, vId, t.v[0], t.v[1], t.v[2]);
    }
}

void Mesh::insertTetrahedronGrid(
        const glm::ivec3& cId, int v0, int v1, int v2, int v3)
{
    Tetrahedron* tet = new Tetrahedron(v0, v1, v2, v3);

    // Literally insert in the grid and mesh
    grid[cId.z][cId.y][cId.x].tetra.insert(tet);
    vert[tet->v[0]].tetra.insert(tet);
    vert[tet->v[1]].tetra.insert(tet);
    vert[tet->v[2]].tetra.insert(tet);
    vert[tet->v[3]].tetra.insert(tet);

    // Set owner cell
    tet->cId = cId;

    // Initialise algo flag
    tet->flag = false;

    // Compute tetrahedron circumcircle
    glm::dvec3 A = vert[tet->v[0]].p;
    glm::dvec3 B = vert[tet->v[1]].p;
    glm::dvec3 C = vert[tet->v[2]].p;
    glm::dvec3 D = vert[tet->v[3]].p;
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


    // Check tetrahedron volume positivness
    glm::dvec3 d10(A - B);
    glm::dvec3 d12(C - B);
    glm::dvec3 d23(D - C);
    glm::dmat3 triple(d10, d12, d23);
    if(glm::determinant(triple) < 0)
    {
        std::swap(tet->v[2], tet->v[3]);
    }
}

void Mesh::removeTetrahedronGrid(Tetrahedron* tet)
{
    grid[tet->cId.z][tet->cId.y][tet->cId.x].tetra.erase(tet);
    vert[tet->v[0]].tetra.erase(tet);
    vert[tet->v[1]].tetra.erase(tet);
    vert[tet->v[2]].tetra.erase(tet);
    vert[tet->v[3]].tetra.erase(tet);
    delete tet;
}

void Mesh::tearDownGrid()
{
    // Collect tetrahedrons
    tetra.clear();
    for(int k=0; k<gridSize.z; ++k)
    {
        for(int j=0; j<gridSize.y; ++j)
        {
            for(int i=0; i<gridSize.x; ++i)
            {
                unordered_set<Tetrahedron*>& tetSet = grid[k][j][i].tetra;
                tetra.insert(tetra.begin(), tetSet.begin(), tetSet.end());
            }
            grid[k][j].clear();
        }
        grid[k].clear();
    }
    grid.clear();


    // Reset vertices adjacency lists
    for(auto& v : vert)
    {
        v.tetra.clear();
    }

    // Reset cached data structures
    _baseQueue.clear();
    _ballQueue.clear();
    _ballPreserved.clear();
    _ballTouched.clear();
}
