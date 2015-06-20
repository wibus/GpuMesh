#ifndef GPUMESH_MESH
#define GPUMESH_MESH

#include <vector>

#include <GLM/glm.hpp>


class Mesh
{
public:

    unsigned int vertCount() const;
    unsigned int elemCount() const;

    double tetrahedronQuality(const glm::ivec4& tet);

    void compileTetrahedronQuality(
            double& qualityMean,
            double& qualityVar);

    void compileFacesAttributes(
            const glm::dvec4& cutPlaneEq,
            std::vector<glm::vec3>& vertices,
            std::vector<signed char>& normals,
            std::vector<unsigned char>& triEdges,
            std::vector<unsigned char>& qualities);


    std::vector<glm::dvec4> vert;
    std::vector<glm::ivec4> tetra;


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
