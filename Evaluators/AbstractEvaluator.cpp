#include "AbstractEvaluator.h"

#include <iostream>
using namespace std;

#include <CellarWorkbench/Misc/Log.h>

using namespace cellar;


struct GpuQual
{
    GLuint min;
    GLuint mean;
};


const double AbstractEvaluator::MAX_INTEGER_VALUE = 4294967296.0;
const double AbstractEvaluator::MIN_QUALITY_PRECISION_DENOM = 256.0;


AbstractEvaluator::AbstractEvaluator(const std::string& shapeMeasuresShader) :
    _initialized(false),
    _computeSimultaneously(true),
    _shapeMeasuresShader(shapeMeasuresShader)
{

}

AbstractEvaluator::~AbstractEvaluator()
{

}

void AbstractEvaluator::evaluateCpuMeshQuality(
        const Mesh& mesh,
        double& minQuality,
        double& qualityMean)
{
    int tetCount = mesh.tetra.size();
    int priCount = mesh.prism.size();
    int hexCount = mesh.hexa.size();

    int elemCount = tetCount + priCount + hexCount;
    std::vector<double> qualities(elemCount);
    int idx = 0;

    for(int i=0; i < tetCount; ++i, ++idx)
        qualities[idx] = tetrahedronQuality(mesh, mesh.tetra[i]);

    for(int i=0; i < priCount; ++i, ++idx)
        qualities[idx] = prismQuality(mesh, mesh.prism[i]);

    for(int i=0; i < hexCount; ++i, ++idx)
        qualities[idx] = hexahedronQuality(mesh, mesh.hexa[i]);


    minQuality = 1.0;
    qualityMean = 0.0;
    for(int i=0; i < elemCount; ++i)
    {
        double qual = qualities[i];

        if(qual < minQuality)
            minQuality = qual;

        qualityMean = (qualityMean * i + qual) / (i + 1);
    }
}

void AbstractEvaluator::evaluateGpuMeshQuality(
        const Mesh& mesh,
        double& minQuality,
        double& qualityMean)
{
    if(!_initialized)
    {
        initializeProgram();

        _initialized = true;
    }


    const size_t polyTypeCount = 3;
    const size_t workgroupSize = 256;
    size_t maxGroupParticipants = workgroupSize * polyTypeCount;
    double maxIntQual = MAX_INTEGER_VALUE / (maxGroupParticipants);
    assert(maxIntQual >= MIN_QUALITY_PRECISION_DENOM);

    size_t tetCount = mesh.tetra.size();
    size_t priCount = mesh.prism.size();
    size_t hexCount = mesh.hexa.size();
    size_t maxSize = glm::max(glm::max(tetCount, priCount), hexCount);
    size_t workgroupCount = ceil(maxSize / (double)workgroupSize);


    GLuint vertSsbo = mesh.glBuffer(EMeshBuffer::VERT);
    GLuint qualSsbo = mesh.glBuffer(EMeshBuffer::QUAL);
    GLuint tetSsbo = mesh.glBuffer(EMeshBuffer::TET);
    GLuint priSsbo = mesh.glBuffer(EMeshBuffer::PRI);
    GLuint hexSsbo = mesh.glBuffer(EMeshBuffer::HEX);

    std::vector<GpuQual> qualBuff(workgroupCount);
    size_t qualSize = sizeof(decltype(qualBuff.front())) * qualBuff.size();
    for(int i=0; i < workgroupCount; ++i)
    {
        qualBuff[i].min = maxIntQual;
        qualBuff[i].mean = 0;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, qualSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, qualSize, qualBuff.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tetSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, priSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, hexSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, qualSsbo);


    // Simulataneous and specialized series gives about the same performance
    // Specialized series gives a tiny, not stable speed boost.
    // (tested on a parametric pri/hex mesh)
    if(_computeSimultaneously)
    {
        _simultaneousProgram.pushProgram();
        _simultaneousProgram.setFloat("MaxQuality", maxIntQual);
        _simultaneousProgram.setInt("TetCount", mesh.tetra.size());
        _simultaneousProgram.setInt("PriCount", mesh.prism.size());
        _simultaneousProgram.setInt("HexCount", mesh.hexa.size());
        glDispatchCompute(workgroupCount, 1, 1);
        _simultaneousProgram.popProgram();
    }
    else
    {
        if(tetCount > 0)
        {
            _tetProgram.pushProgram();
            _tetProgram.setInt("TetCount", tetCount);
            _tetProgram.setFloat("MaxQuality", maxIntQual);
            glDispatchCompute(ceil(tetCount / (double)workgroupSize), 1, 1);
            _tetProgram.popProgram();
        }

        if(priCount > 0)
        {
            _priProgram.pushProgram();
            _priProgram.setInt("PriCount", priCount);
            _priProgram.setFloat("MaxQuality", maxIntQual);
            glDispatchCompute(ceil(priCount / (double)workgroupSize), 1, 1);
            _priProgram.popProgram();
        }

        if(hexCount > 0)
        {
            _hexProgram.pushProgram();
            _hexProgram.setInt("HexCount", hexCount);
            _hexProgram.setFloat("MaxQuality", maxIntQual);
            glDispatchCompute(ceil(hexCount / (double)workgroupSize), 1, 1);
            _hexProgram.popProgram();
        }
    }

    // Fetch workgroup's statistics from GPU
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, qualSsbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, qualSize, qualBuff.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    // Combine workgroup's statistics
    minQuality = 1.0;
    qualityMean = 0.0;
    double meanWeight = 0.0;
    for(size_t i=0, polyId=0; i < workgroupCount; ++i, polyId+=workgroupSize)
    {
        minQuality = glm::min(minQuality, qualBuff[i].min / maxIntQual);

        size_t groupTet = (tetCount > polyId) ? glm::min(tetCount - polyId, workgroupSize) : 0;
        size_t groupPri = (priCount > polyId) ? glm::min(priCount - polyId, workgroupSize) : 0;
        size_t groupHex = (hexCount > polyId) ? glm::min(hexCount - polyId, workgroupSize) : 0;
        double groupParticipants = (groupTet + groupPri + groupHex);
        double groupWeight = groupParticipants / maxGroupParticipants;
        double groupMean = (qualBuff[i].mean / MAX_INTEGER_VALUE);

        // 'groupMean' is implicitly multiplied by 'groupWeight'
        qualityMean = (qualityMean*meanWeight + groupMean) /
                      (meanWeight + groupWeight);
        meanWeight += groupWeight;
    }
}

void AbstractEvaluator::initializeProgram()
{
    getLog().postMessage(new Message('I', false,
        "Initializing insphere quality evaluator compute shader",
        "AbstractEvaluator"));


    // Simultenous evalution shader
    _simultaneousProgram.addShader(GL_COMPUTE_SHADER, _shapeMeasuresShader);
    _simultaneousProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/Measuring/SimultaneousEvaluation.glsl");
    _simultaneousProgram.link();


    // Specialized evaluation shader series
    _tetProgram.addShader(GL_COMPUTE_SHADER, _shapeMeasuresShader);
    _tetProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/Measuring/TetrahedraEvaluation.glsl");
    _tetProgram.link();

    _priProgram.addShader(GL_COMPUTE_SHADER, _shapeMeasuresShader);
    _priProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/Measuring/PrismsEvaluation.glsl");
    _priProgram.link();

    _hexProgram.addShader(GL_COMPUTE_SHADER, _shapeMeasuresShader);
    _hexProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/Measuring/HexahedraEvaluation.glsl");
    _hexProgram.link();
}
