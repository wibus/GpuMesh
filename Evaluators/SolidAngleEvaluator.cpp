#include "SolidAngleEvaluator.h"


SolidAngleEvaluator::SolidAngleEvaluator() :
    AbstractEvaluator(":/shaders/compute/Quality/SolidAngle.glsl")
{

}

SolidAngleEvaluator::~SolidAngleEvaluator()
{

}

double SolidAngleEvaluator::tetrahedronQuality(
        const Mesh& mesh, const MeshTet& tet) const
{
    const glm::dvec3& t0 = mesh.vert[tet.v[0]];
    const glm::dvec3& t1 = mesh.vert[tet.v[1]];
    const glm::dvec3& t2 = mesh.vert[tet.v[2]];
    const glm::dvec3& t3 = mesh.vert[tet.v[3]];

    double q0 = solidAngle(t1 - t0, t2 - t0, t3 - t0);
    double q1 = solidAngle(t0 - t1, t2 - t1, t3 - t1);
    double q2 = solidAngle(t0 - t2, t1 - t2, t3 - t2);
    double q3 = solidAngle(t0 - t3, t1 - t3, t2 - t3);

    double minQ = glm::min(glm::min(q0, q1),
                           glm::min(q2, q3));
    return minQ * 3.67423461417; // 9 / sqrt(6)
}

double SolidAngleEvaluator::prismQuality(
        const Mesh& mesh, const MeshPri& pri) const
{
    const glm::dvec3& t0 = mesh.vert[pri.v[0]];
    const glm::dvec3& t1 = mesh.vert[pri.v[1]];
    const glm::dvec3& t2 = mesh.vert[pri.v[2]];
    const glm::dvec3& t3 = mesh.vert[pri.v[3]];
    const glm::dvec3& t4 = mesh.vert[pri.v[4]];
    const glm::dvec3& t5 = mesh.vert[pri.v[5]];

    double q0 = solidAngle(t1 - t0, t2 - t0, t4 - t0);
    double q1 = solidAngle(t0 - t1, t3 - t1, t5 - t1);
    double q2 = solidAngle(t0 - t2, t3 - t2, t4 - t2);
    double q3 = solidAngle(t1 - t3, t2 - t3, t5 - t3);
    double q4 = solidAngle(t0 - t4, t2 - t4, t5 - t4);
    double q5 = solidAngle(t1 - t5, t3 - t5, t4 - t5);

    double minQ = glm::min(glm::min(q0, q1),
                           glm::min(glm::min(q2, q3),
                                    glm::min(q4, q5)));
    return minQ * 2.61312592975; // 1.0 / <max val for regular prism>
}

double SolidAngleEvaluator::hexahedronQuality(
        const Mesh& mesh, const MeshHex& hex) const
{
    const glm::dvec3& t0 = mesh.vert[hex.v[0]];
    const glm::dvec3& t1 = mesh.vert[hex.v[1]];
    const glm::dvec3& t2 = mesh.vert[hex.v[2]];
    const glm::dvec3& t3 = mesh.vert[hex.v[3]];
    const glm::dvec3& t4 = mesh.vert[hex.v[4]];
    const glm::dvec3& t5 = mesh.vert[hex.v[5]];
    const glm::dvec3& t6 = mesh.vert[hex.v[6]];
    const glm::dvec3& t7 = mesh.vert[hex.v[7]];

    double q0 = solidAngle(t1 - t0, t2 - t0, t4 - t0);
    double q1 = solidAngle(t0 - t1, t3 - t1, t5 - t1);
    double q2 = solidAngle(t0 - t2, t3 - t2, t6 - t2);
    double q3 = solidAngle(t1 - t3, t2 - t3, t7 - t3);
    double q4 = solidAngle(t0 - t4, t5 - t4, t6 - t4);
    double q5 = solidAngle(t1 - t5, t4 - t5, t7 - t5);
    double q6 = solidAngle(t2 - t6, t4 - t6, t7 - t6);
    double q7 = solidAngle(t3 - t7, t5 - t7, t6 - t7);

    double minQ = glm::min(glm::min(glm::min(q0, q1), glm::min(q2, q3)),
                           glm::min(glm::min(q4, q5), glm::min(q6, q7)));
    return minQ * 1.41421356237; // sqrt(2)
}

inline double SolidAngleEvaluator::solidAngle(
        const glm::dvec3& a,
        const glm::dvec3& b,
        const glm::dvec3& c) const
{
    double al = glm::length(a);
    double bl = glm::length(b);
    double cl = glm::length(c);

    return glm::abs(glm::determinant(glm::dmat3(a, b, c))) /
            sqrt( 2.0 * (al*bl + glm::dot(a, b)) *
                        (bl*cl + glm::dot(b, c)) *
                        (cl*al + glm::dot(c, a)));
}
