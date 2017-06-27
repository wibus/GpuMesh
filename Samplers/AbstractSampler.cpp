#include "AbstractSampler.h"

#include <GLM/gtc/constants.hpp>
#include <GLM/gtc/matrix_transform.hpp>

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"


void setCudaMetricScaling(double scaling);
void setCudaMetricScalingSqr(double scalingSqr);
void setCudaMetricScalingCube(double scalingCube);
void setCudaMetricAspectRatio(double aspectRatio);
void setCudaRotMat(const glm::dmat3& rotMat, const glm::dmat3& rotInv);

AbstractSampler::AbstractSampler(
        const std::string& name,
        const std::string& shader,
        const installCudaFct installCuda) :
    _scaling(1.0),
    _scaling2(1.0),
    _scaling3(1.0),
    _aspectRatio(1.0),
    _discretizationDepth(-1),
    _samplingName(name),
    _samplingShader(shader),
    _baseShader(":/glsl/compute/Sampling/Base.glsl"),
    _installCuda(installCuda)
{
    _rotMat = glm::dmat3(glm::rotate(
                glm::dmat4(_rotMat),
                glm::pi<double>() / 4.0,
                glm::dvec3(0, 0, 1)));

    _rotMat = glm::dmat3(glm::rotate(
                glm::dmat4(_rotMat),
                glm::pi<double>() / 4.0,
                glm::dvec3(0, -1, 0)));

    _rotInv = glm::transpose(_rotMat);
}

AbstractSampler::~AbstractSampler()
{

}

void AbstractSampler::setScaling(double scaling)
{
    _scaling = scaling;
    _scaling2 = _scaling * scaling;
    _scaling3 = _scaling2 * scaling;
}

void AbstractSampler::setAspectRatio(double ratio)
{
    _aspectRatio = ratio;
}

void AbstractSampler::setDiscretizationDepth(int depth)
{
    _discretizationDepth = depth;
}

std::string AbstractSampler::samplingShader() const
{
    return _samplingShader;
}

void AbstractSampler::installPlugin(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    if(!_samplingShader.empty())
    {
        program.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            _baseShader.c_str()
        });

        program.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            _samplingShader.c_str()
        });
    }

    _installCuda();
}

void AbstractSampler::setPluginGlslUniforms(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{
    program.setFloat("MetricScaling", scaling());
    program.setFloat("MetricScalingSqr", scalingSqr());
    program.setFloat("MetricScalingCube", scalingCube());
    program.setFloat("MetricAspectRatio", aspectRatio());
    program.setMat3f("RotMat", glm::mat3(_rotMat));
    program.setMat3f("RotInv", glm::mat3(_rotInv));
}

void AbstractSampler::setPluginCudaUniforms(
        const Mesh& mesh) const
{
    setCudaMetricScaling(scaling());
    setCudaMetricScalingSqr(scalingSqr());
    setCudaMetricScalingCube(scalingCube());
    setCudaMetricAspectRatio(aspectRatio());
    setCudaRotMat(_rotMat, _rotInv);
}

void AbstractSampler::updateGlslData(const Mesh& mesh) const
{

}

void AbstractSampler::updateCudaData(const Mesh& mesh) const
{

}

void AbstractSampler::clearGlslMemory(const Mesh& mesh) const
{

}

void AbstractSampler::clearCudaMemory(const Mesh& mesh) const
{

}

MeshMetric AbstractSampler::interpolateMetrics(
        const MeshMetric& m1,
        const MeshMetric& m2,
        double a) const
{
    return glm::mix(m1, m2, a);
}

MeshMetric AbstractSampler::vertMetric(const Mesh& mesh, unsigned int vId) const
{
    return vertMetric(mesh.verts[vId].p);
}

inline MeshMetric atanXY(double scaling, double ratio, const glm::dvec3& position)
{
    double s2 = scaling * scaling;
    glm::dvec3 vp = position * 4.0;
    double x = vp.x, y = vp.y, z = vp.z;

    double x2_p_1 = x*x + 1;
    double y2_p_1 = y*y + 1;

    double rxx = 2 * x * atan(y) / (x2_p_1*x2_p_1);
    double rxy = 1 / (x2_p_1 * y2_p_1);
    double ryy = 2 * y * atan(x) / (y2_p_1*y2_p_1);

    double rxz = 0.0;
    double ryz = 0.0;
    double rzz = s2;

    double T = rxx + ryy;
    double D = rxx*ryy - ryz*ryz;
    double discr = sqrt(T*T/4 - D);
    double l0 = T/2 + discr;
    double l1 = T/2 - discr;

    glm::dvec2 v0(1, 0);
    glm::dvec2 v1(0, 1);
    if(rxy != 0.0)
    {
        v0 = glm::normalize(glm::dvec2(l0-ryy, rxy));
        v1 = glm::normalize(glm::dvec2(l1-ryy, rxy));
    }
    glm::dmat2 R(v0, v1);

    l0 = glm::min(glm::max(l0*l0 * s2 * s2, s2), 1.0e9);
    l1 = glm::min(glm::max(l1*l1 * s2 * s2, s2), 1.0e9);

    glm::dmat2 M_abs = glm::transpose(R) * glm::dmat2(l0, 0, 0, l1) * R;

    return MeshMetric(
        glm::dvec3(M_abs[0][0], M_abs[0][1], rxz),
        glm::dvec3(M_abs[1][0], M_abs[1][1], ryz),
        glm::dvec3(rxz,         ryz,         rzz));
}


MeshMetric AbstractSampler::vertMetric(const glm::dvec3& position) const
{
    double x = (_rotMat * position).x * (2.5 * glm::pi<double>());

    double a = aspectRatio();
    double c = (1.0 - glm::cos(x)) / 2.0;
    double sizeX = scaling() * glm::pow(a, glm::pow(c, a));

    double Mx = sizeX * sizeX;
    double My = scalingSqr();
    double Mz = My;

    glm::dmat3 M = MeshMetric(
        glm::dvec3(Mx, 0,  0),
        glm::dvec3(0,  My, 0),
        glm::dvec3(0,  0,  Mz));

    return _rotInv * M * _rotMat;
}

void AbstractSampler::boundingBox(
        const Mesh& mesh,
        glm::dvec3& minBounds,
        glm::dvec3& maxBounds) const
{
    minBounds = glm::dvec3(INFINITY);
    maxBounds = glm::dvec3(-INFINITY);
    size_t vertCount = mesh.verts.size();
    for(size_t v=0; v < vertCount; ++v)
    {
        const glm::dvec3& vertPos = mesh.verts[v].p;
        minBounds = glm::min(minBounds, vertPos);
        maxBounds = glm::max(maxBounds, vertPos);
    }
}
