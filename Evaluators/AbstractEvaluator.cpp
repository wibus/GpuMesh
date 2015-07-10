#include "AbstractEvaluator.h"

#include <iostream>
#include <sstream>

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/GpuMesh.h"

using namespace cellar;
using namespace std;



struct GpuQual
{
    GLuint min;
    GLuint mean;

    inline GpuQual() : min(UINT32_MAX), mean(0) {}
};


const double AbstractEvaluator::VALIDITY_EPSILON = 1e-6;
const double AbstractEvaluator::MAX_INTEGER_VALUE = 4294967296.0;
const double AbstractEvaluator::MIN_QUALITY_PRECISION_DENOM = 256.0;

AbstractEvaluator::AbstractEvaluator(const std::string& shapeMeasuresShader) :
    _initialized(false),
    _computeSimultaneously(true),
    _shapeMeasuresShader(shapeMeasuresShader),
    _qualSsbo(0)
{

}

AbstractEvaluator::~AbstractEvaluator()
{
    glDeleteBuffers(1, &_qualSsbo);
}


bool AbstractEvaluator::assessMeasureValidy()
{
    Mesh mesh;
    mesh.vert.push_back(glm::dvec3(0, 0, 0));
    mesh.vert.push_back(glm::dvec3(1, 0, 0));
    mesh.vert.push_back(glm::dvec3(0.5, sqrt(3)/6.0, sqrt(2.0/3)));
    mesh.vert.push_back(glm::dvec3(0.5, sqrt(3)/2.0, 0));

    mesh.vert.push_back(glm::dvec3(0, 0, 0));
    mesh.vert.push_back(glm::dvec3(1, 0, 0));
    mesh.vert.push_back(glm::dvec3(0, 1, 0));
    mesh.vert.push_back(glm::dvec3(1, 1, 0));
    mesh.vert.push_back(glm::dvec3(0, sqrt(3)/2, sqrt(3)/2));
    mesh.vert.push_back(glm::dvec3(1, sqrt(3)/2, sqrt(3)/2));

    mesh.vert.push_back(glm::dvec3(0, 0, 0));
    mesh.vert.push_back(glm::dvec3(1, 0, 0));
    mesh.vert.push_back(glm::dvec3(0, 1, 0));
    mesh.vert.push_back(glm::dvec3(1, 1, 0));
    mesh.vert.push_back(glm::dvec3(0, 0, 1));
    mesh.vert.push_back(glm::dvec3(1, 0, 1));
    mesh.vert.push_back(glm::dvec3(0, 1, 1));
    mesh.vert.push_back(glm::dvec3(1, 1, 1));

    const MeshTet tet = MeshTet(0, 1, 2, 3);
    const MeshPri pri = MeshPri(4, 5, 6, 7, 8, 9);
    const MeshHex hex = MeshHex(10, 11, 12, 13, 14, 15, 16, 17);
    double regularTet = tetrahedronQuality(mesh, tet);
    double regularPri = prismQuality(mesh, pri);
    double regularHex = hexahedronQuality(mesh, hex);

    if(glm::abs(regularTet - 1.0) < VALIDITY_EPSILON &&
       glm::abs(regularPri - 1.0) < VALIDITY_EPSILON &&
       glm::abs(regularHex - 1.0) < VALIDITY_EPSILON)
    {
        getLog().postMessage(new Message('I', false,
            "Quality evaluator's measure is valid.", "AbstractEvaluator"));
        return true;
    }
    else
    {
        stringstream log;
        log.precision(20);
        log << "Quality evaluator's measure is invalid." << endl;
        log << "Regular tetrahedron quality: " << regularTet << endl;
        log << "Regular prism quality: " << regularPri << endl;
        log << "Regular hexahedron quality: " << regularHex << endl;
        getLog().postMessage(new Message('E', true, log.str(), "AbstractEvaluator"));

        assert(glm::abs(regularTet - 1.0) < VALIDITY_EPSILON &&
               glm::abs(regularPri - 1.0) < VALIDITY_EPSILON &&
               glm::abs(regularHex - 1.0) < VALIDITY_EPSILON);
        return false;
    }
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
        initializeProgram(mesh);

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


    std::vector<GpuQual> qualBuff(workgroupCount);
    size_t qualSize = sizeof(decltype(qualBuff.front())) * qualBuff.size();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _qualSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, qualSize, qualBuff.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    mesh.bindShaderStorageBuffers();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _qualSsbo);


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
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _qualSsbo);
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

void AbstractEvaluator::initializeProgram(const Mesh& mesh)
{
    getLog().postMessage(new Message('I', false,
        "Initializing evaluator compute shader", "AbstractEvaluator"));

    std::vector<std::string> shapeMeasure = {
        mesh.meshGeometryShaderName(),
        _shapeMeasuresShader
    };


    // Simultenous evalution shader
    _simultaneousProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _simultaneousProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/SimultaneousEvaluation.glsl"});
    _simultaneousProgram.link();
    mesh.uploadGeometry(_simultaneousProgram);


    // Specialized evaluation shader series
    _tetProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _tetProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/TetrahedraEvaluation.glsl"});
    _tetProgram.link();
    mesh.uploadGeometry(_tetProgram);

    _priProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _priProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/PrismsEvaluation.glsl"});
    _priProgram.link();
    mesh.uploadGeometry(_priProgram);

    _hexProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _hexProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/HexahedraEvaluation.glsl"});
    _hexProgram.link();
    mesh.uploadGeometry(_hexProgram);

    // Shader storage quality blocks
    glGenBuffers(1, &_qualSsbo);
}
