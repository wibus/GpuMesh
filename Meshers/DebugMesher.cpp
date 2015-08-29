#include "DebugMesher.h"

#include <GLM/gtc/random.hpp>

using namespace std;


DebugMesher::DebugMesher()
{
    using namespace std::placeholders;
    _modelFuncs.setDefault("Singles");
    _modelFuncs.setContent({
        {string("Singles"), ModelFunc(bind(&DebugMesher::genSingles,   this, _1, _2))},
        {string("Squish"),  ModelFunc(bind(&DebugMesher::genSquish, this, _1, _2))},
    });
}

DebugMesher::~DebugMesher()
{

}

void DebugMesher::genSingles(Mesh& mesh, size_t vertexCount)
{
    // Tetrahedron
    mesh.tets.push_back(MeshTet(0, 1, 2, 3));
    mesh.verts.push_back(glm::dvec3(0, 0, 0));
    mesh.verts.push_back(glm::dvec3(1, 0, 0));
    mesh.verts.push_back(glm::dvec3(0.5, sqrt(3.0)/6, sqrt(2.0/3)));
    mesh.verts.push_back(glm::dvec3(0.5, sqrt(3.0)/2, 0));
    for(int i=0; i<4; ++i)
    {
        mesh.verts[i].p += glm::linearRand(glm::dvec3(-0.5), glm::dvec3(0.5));
        mesh.verts[i].p *= 0.5;
        mesh.verts[i].p += glm::dvec3(-0.75, 0.0, 0.0);
    }


    // Prism
    mesh.pris.push_back(MeshPri(4, 5, 6, 7, 8, 9));
    mesh.verts.push_back(glm::dvec3(0, 0, 0));
    mesh.verts.push_back(glm::dvec3(1, 0, 0));
    mesh.verts.push_back(glm::dvec3(0, 1, 0));
    mesh.verts.push_back(glm::dvec3(1, 1, 0));
    mesh.verts.push_back(glm::dvec3(0, 0.5, sqrt(3.0)/2));
    mesh.verts.push_back(glm::dvec3(1, 0.5, sqrt(3.0)/2));
    for(int i=4; i<10; ++i)
    {
        mesh.verts[i].p += glm::linearRand(glm::dvec3(-0.5), glm::dvec3(0.5));
        mesh.verts[i].p *= 0.5;
        mesh.verts[i].p += glm::dvec3(0.0);
    }


    // Hexahedron
    mesh.hexs.push_back(MeshHex(10, 11, 12, 13, 14, 15, 16, 17));
    mesh.verts.push_back(glm::dvec3(0, 0, 0));
    mesh.verts.push_back(glm::dvec3(1, 0, 0));
    mesh.verts.push_back(glm::dvec3(0, 1, 0));
    mesh.verts.push_back(glm::dvec3(1, 1, 0));
    mesh.verts.push_back(glm::dvec3(0, 0, 1));
    mesh.verts.push_back(glm::dvec3(1, 0, 1));
    mesh.verts.push_back(glm::dvec3(0, 1, 1));
    mesh.verts.push_back(glm::dvec3(1, 1, 1));
    for(int i=10; i<18; ++i)
    {
        mesh.verts[i].p += glm::linearRand(glm::dvec3(-0.5), glm::dvec3(0.5));
        mesh.verts[i].p *= 0.5;
        mesh.verts[i].p += glm::dvec3(0.75, 0.0, 0.0);
    }
}

void DebugMesher::genSquish(Mesh& mesh, size_t vertexCount)
{
    double squishRadius = 0.3;
    double squishHeight = 0.6;

    const int pow1_3 =  glm::pow((double)vertexCount, 1.0/3.0);
    const int pow1_3_pair = ((pow1_3 + 1) / 2) * 2;
    const int X_COUNT = pow1_3_pair;
    const int Y_COUNT = pow1_3_pair;
    const int Z_COUNT = pow1_3_pair;
    for(int z=-Z_COUNT/2; z <= Z_COUNT/2; ++z)
    {
        for(int y=-Y_COUNT/2; y <= Y_COUNT/2; ++y)
        {
            for(int x=-X_COUNT/2; x <= X_COUNT/2; ++x)
            {
                glm::dvec2 arm;
                double radius = glm::max(glm::abs(x), glm::abs(y)) * (2.0 * squishRadius / Y_COUNT);
                if(radius != 0.0) arm = glm::normalize(glm::dvec2(x, y)) * radius;

                mesh.verts.push_back(glm::dvec3(
                    arm, z * squishHeight / Z_COUNT));
            }
        }
    }

    const int X_WIDTH = 1;
    const int Y_WIDTH = X_COUNT + 1;
    const int Z_WIDTH = (Y_COUNT+1) * Y_WIDTH;
    for(int z=0; z < Z_COUNT; ++z)
    {
        for(int y=0; y< Y_COUNT; ++y)
        {
            for(int x=0; x< X_COUNT; ++x)
            {
                MeshHex hex(
                    (x+0) * X_WIDTH + (y+0) * Y_WIDTH + (z+0) * Z_WIDTH,
                    (x+1) * X_WIDTH + (y+0) * Y_WIDTH + (z+0) * Z_WIDTH,
                    (x+0) * X_WIDTH + (y+1) * Y_WIDTH + (z+0) * Z_WIDTH,
                    (x+1) * X_WIDTH + (y+1) * Y_WIDTH + (z+0) * Z_WIDTH,
                    (x+0) * X_WIDTH + (y+0) * Y_WIDTH + (z+1) * Z_WIDTH,
                    (x+1) * X_WIDTH + (y+0) * Y_WIDTH + (z+1) * Z_WIDTH,
                    (x+0) * X_WIDTH + (y+1) * Y_WIDTH + (z+1) * Z_WIDTH,
                    (x+1) * X_WIDTH + (y+1) * Y_WIDTH + (z+1) * Z_WIDTH);
                mesh.hexs.push_back(hex);
            }
        }
    }
}
