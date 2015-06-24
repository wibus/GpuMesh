#ifndef GPUMESH_MESH
#define GPUMESH_MESH

#include <vector>
#include <functional>

#include <GLM/glm.hpp>


struct MeshVert
{
    glm::dvec3 p;

    MeshVert() : p(0) {}
    MeshVert(const glm::dvec3 p) : p(p){}
    inline double& operator[] (int c) { return p[c]; }
    inline const double& operator[] (int c) const { return p[c]; }
    inline operator glm::dvec3() const { return p; }
};

struct MeshTri
{
    int v[3];

    MeshTri() : v{-1, -1, -1} {}
    MeshTri(int v0, int v1, int v2) : v{v0, v1, v2} {}
    inline int& operator[] (int i) { return v[i]; }
    inline const int& operator[] (int i) const { return v[i]; }
};


struct MeshTet
{
    int v[4];

    MeshTet() : v{-1, -1, -1, -1} {}
    MeshTet(int v0, int v1, int v2, int v3) : v{v0, v1, v2, v3} {}
    inline int& operator[] (int i) { return v[i]; }
    inline const int& operator[] (int i) const { return v[i]; }

    static const int FACE_COUNT = 4;
    static const MeshTri faces[FACE_COUNT];
    static const int EDGE_COUNT = 6;
    static const int edges[EDGE_COUNT][2];
};


struct MeshPri
{
    int v[6];

    MeshPri() : v{-1, -1, -1, -1, -1, -1} {}
    MeshPri(int v0, int v1, int v2,
            int v3, int v4, int v5) :
        v{v0, v1, v2, v3, v4, v5} {}
    inline int operator[] (int i) { return v[i]; }
    inline const int& operator[] (int i) const { return v[i]; }

    static const int FACE_COUNT = 8;
    static const MeshTri faces[FACE_COUNT];
    static const int EDGE_COUNT = 9;
    static const int edges[EDGE_COUNT][2];
};


struct MeshHex
{
    int v[8];

    MeshHex() : v{-1, -1, -1, -1, -1, -1, -1, -1} {}
    MeshHex(int v0, int v1, int v2, int v3,
            int v4, int v5, int v6, int v7) :
        v{v0, v1, v2, v3, v4, v5, v6, v7} {}
    inline int operator[] (int i) { return v[i]; }
    inline const int& operator[] (int i) const { return v[i]; }

    static const int FACE_COUNT = 12;
    static const MeshTri faces[FACE_COUNT];
    static const int EDGE_COUNT = 12;
    static const int edges[EDGE_COUNT][2];
};


typedef std::function<glm::dvec3(const glm::dvec3&)>
    BoundaryCallback;

struct MeshVertProperties
{
    bool isFixed;
    std::vector<int> neighbors;

    bool isBoundary;
    BoundaryCallback boundaryCallback;

    MeshVertProperties();
    MeshVertProperties(bool isFixed);
    MeshVertProperties(const BoundaryCallback& boundaryCallback);
};


class Mesh
{
public:

    unsigned int vertCount() const;
    unsigned int elemCount() const;

    void clear();

    double tetrahedronQuality(const MeshTet& tet);
    double hexahedronQuality(const MeshHex& hex);
    double prismQuality(const MeshPri& pri);

    void compileElementQuality(
            double& qualityMean,
            double& qualityVar,
            double& minQuality);

    void compileVertexAdjacency();

    void compileFacesAttributes(
            const glm::dvec4& cutPlaneEq,
            std::vector<glm::vec3>& vertices,
            std::vector<signed char>& normals,
            std::vector<unsigned char>& triEdges,
            std::vector<unsigned char>& qualities);


    std::vector<MeshVert> vert;
    std::vector<MeshTet> tetra;
    std::vector<MeshPri> prism;
    std::vector<MeshHex> hexa;

    std::vector<MeshVertProperties> vertProperties;


protected:
    void pushTriangle(
            std::vector<glm::vec3>& vertices,
            std::vector<signed char>& normals,
            std::vector<unsigned char>& triEdges,
            std::vector<unsigned char>& qualities,
            const glm::dvec3& A,
            const glm::dvec3& B,
            const glm::dvec3& C,
            const glm::dvec3& n,
            bool fromQuad,
            double quality);

    virtual void addEdge(int firstVert,
                         int secondVert);
};




// IMPLEMENTATION //
inline unsigned int Mesh::vertCount() const
{
    return vert.size();
}

inline unsigned int Mesh::elemCount() const
{
    return tetra.size() + prism.size() + hexa.size();
}

#endif // GPUMESH_MESH