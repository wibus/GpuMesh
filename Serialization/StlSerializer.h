#ifndef GPUMESH_STLSERIALIZER
#define GPUMESH_STLSERIALIZER

#include <fstream>

#include <GLM/glm.hpp>

#include "AbstractSerializer.h"


class StlSerializer : public AbstractSerializer
{
public:
    StlSerializer();
    virtual ~StlSerializer();

    virtual bool serialize(
            const std::string& fileName,
            const MeshCrew& crew,
            const Mesh& mesh) const override;

protected:
    static void headerAscii(std::ofstream& stream, const Mesh& mesh);
    static void headerBin(std::ofstream& stream, const Mesh& mesh);

    static void footerAscii(std::ofstream& stream, const Mesh& mesh);
    static void footerBin(std::ofstream& stream, const Mesh& mesh);

    static void printAscii(std::ofstream& stream, const glm::vec3 v[], double qual);
    static void printFlatBin(std::ofstream& stream, const glm::vec3 v[], double qual);
    static void printColorBin(std::ofstream& stream, const glm::vec3 v[], double qual);
};

#endif // GPUMESH_STLSERIALIZER
