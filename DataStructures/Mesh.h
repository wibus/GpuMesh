#ifndef GPUMESH_MESH
#define GPUMESH_MESH

#include <vector>

#include <GLM/glm.hpp>


struct MeshVert
{
    glm::dvec3 p;
    bool boundary;

    MeshVert() : p(0), boundary(false) {}
    MeshVert(const glm::dvec3 p) : p(p), boundary(false) {}
    MeshVert(const glm::dvec3 p, bool boundary) : p(p), boundary(boundary) {}
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
};

struct MeshPen
{
    int v[6];

    MeshPen() : v{-1, -1, -1, -1, -1, -1} {}
    MeshPen(int v0, int v1, int v2,
            int v3, int v4, int v5) :
        v{v0, v1, v2, v3, v4, v5} {}
    inline int operator[] (int i) { return v[i]; }
    inline const int& operator[] (int i) const { return v[i]; }

    static const int FACE_COUNT = 8;
    static const MeshTri faces[FACE_COUNT];
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
};


class Mesh
{
public:

    unsigned int vertCount() const;
    unsigned int elemCount() const;

    double tetrahedronQuality(const MeshTet& tet);
    double pentahedronQuality(const MeshPen& pen);
    double hexahedronQuality(const MeshHex& hex);

    void compileElementQuality(
            double& qualityMean,
            double& qualityVar);

    void compileFacesAttributes(
            const glm::dvec4& cutPlaneEq,
            std::vector<glm::vec3>& vertices,
            std::vector<signed char>& normals,
            std::vector<unsigned char>& triEdges,
            std::vector<unsigned char>& qualities);


    std::vector<MeshVert> vert;
    std::vector<MeshTet> tetra;
    std::vector<MeshPen> penta;
    std::vector<MeshHex> hexa;


private:
    void pushTriangle(
            std::vector<glm::vec3>& vertices,
            std::vector<signed char>& normals,
            std::vector<unsigned char>& triEdges,
            std::vector<unsigned char>& qualities,
            const glm::dvec3& A,
            const glm::dvec3& B,
            const glm::dvec3& C,
            const glm::dvec3& n,
            double quality);
};


#endif // GPUMESH_MESH
