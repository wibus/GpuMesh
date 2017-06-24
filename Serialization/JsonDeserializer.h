#ifndef GPUMESH_JSONDESERIALIZER
#define GPUMESH_JSONDESERIALIZER

#include <iostream>

#include "AbstractDeserializer.h"
#include "DataStructures/Mesh.h"


class JsonDeserializer : public AbstractDeserializer
{
public:
    JsonDeserializer();
    virtual ~JsonDeserializer();

    virtual bool deserialize(
            const std::string& fileName,
            Mesh& mesh,
            std::vector<MeshMetric>& metrics) const override;

protected:
    bool readTag(std::istream &is, std::string& str) const;
    bool readString(std::istream &is, std::string& str) const;

    bool openArray(std::istream &is) const;
    bool closeArray(std::istream &is) const;

    template<typename T>
    bool readArray(std::istream &is, std::vector<T>& a) const;

};

#endif // GPUMESH_JSONDESERIALIZER
