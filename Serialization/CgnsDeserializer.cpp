#include "CgnsDeserializer.h"

#include "cgnslib.h"


CgnsDeserializer::CgnsDeserializer()
{

}

CgnsDeserializer::~CgnsDeserializer()
{

}

bool CgnsDeserializer::deserialize(
        const std::string& fileName,
        Mesh& mesh) const
{
    int file_index = -1;
    if(cg_open(fileName.c_str(), CG_MODE_READ, &file_index))
    {
        return false;
    }

    cg_close(file_index);

    return true;
}
