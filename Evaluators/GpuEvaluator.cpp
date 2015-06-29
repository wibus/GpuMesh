#include "GpuEvaluator.h"

#include <iostream>

#include "DataStructures/GpuMesh.h"

using namespace std;

const double GpuEvaluator::MAX_INTEGER_VALUE = 4294967296.0;
const double GpuEvaluator::MIN_QUALITY_PRECISION_DENOM = 256.0;

GpuEvaluator::GpuEvaluator() :
    _initialized(false)
{

}

GpuEvaluator::~GpuEvaluator()
{

}

void GpuEvaluator::evaluateMeshQuality(
        const Mesh& mesh,
        double& qualityMean,
        double& qualityVar,
        double& minQuality)
{
    GLuint vertSsbo = mesh.glBuffer(EMeshBuffer::VERT);
    GLuint qualSsbo = mesh.glBuffer(EMeshBuffer::QUAL);
    GLuint tetSsbo = mesh.glBuffer(EMeshBuffer::TET);
    GLuint priSsbo = mesh.glBuffer(EMeshBuffer::PRI);
    GLuint hexSsbo = mesh.glBuffer(EMeshBuffer::HEX);


    size_t tetCount = mesh.tetra.size();
    size_t priCount = mesh.prism.size();
    size_t hexCount = mesh.hexa.size();
    size_t polyCount = tetCount + priCount + hexCount;
    size_t maxSize = glm::max(glm::max(tetCount, priCount), hexCount);

    double maxIntQual = MAX_INTEGER_VALUE / polyCount;
    assert(maxIntQual >= MIN_QUALITY_PRECISION_DENOM);

    GpuQual gpuQual;
    gpuQual.mean = 0;
    gpuQual.var = 0;
    gpuQual.min = maxIntQual;

    if(!_initialized)
    {
        initializeProgram();

        _initialized = true;
    }


    _evaluatorProgram.pushProgram();
    _evaluatorProgram.setFloat("MaxQuality", maxIntQual);
    _evaluatorProgram.setInt("TetCount", mesh.tetra.size());
    _evaluatorProgram.setInt("PriCount", mesh.prism.size());
    _evaluatorProgram.setInt("HexCount", mesh.hexa.size());

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, qualSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, tetSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, priSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, hexSsbo);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, qualSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GpuQual), &gpuQual, GL_STATIC_DRAW);

    glDispatchCompute(ceil(maxSize / 256.0), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    _evaluatorProgram.popProgram();

    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GpuQual), &gpuQual);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    qualityMean = gpuQual.mean / MAX_INTEGER_VALUE;
    qualityVar = gpuQual.var / maxIntQual;
    minQuality = gpuQual.min / maxIntQual;
}

void GpuEvaluator::initializeProgram()
{
    cout << "Initializing insphere quality evaluator compute shader" << endl;
    _evaluatorProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/InsphereQualityMeasures.glsl");
    _evaluatorProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/PolyhedronQualityEvaluator.glsl");
    _evaluatorProgram.link();
}

