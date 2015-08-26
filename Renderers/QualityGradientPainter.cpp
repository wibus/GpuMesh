#include "QualityGradientPainter.h"

#include <GL3/gl3w.h>

#include <CellarWorkbench/Image/Image.h>
#include <CellarWorkbench/Image/ImageBank.h>
#include <CellarWorkbench/GL/GlProgram.h>

using namespace std;
using namespace cellar;


QualityGradientPainter::QualityGradientPainter()
{

}

QualityGradientPainter::~QualityGradientPainter()
{

}

std::string QualityGradientPainter::generate(int width, int height, int minHeight)
{
    Image lutImage(width, height);


    GlProgram lutProg;
    lutProg.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/Measuring/QualityGradient.glsl");
    lutProg.addShader(GL_COMPUTE_SHADER,
        ":/shaders/generic/QualityLut.glsl");
    lutProg.link();
    lutProg.pushProgram();
    lutProg.setFloat("MinHeight", float(minHeight) / height);

    GLuint lutBuff;
    glGenBuffers(1, &lutBuff);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, lutBuff);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        lutImage.dataSize() * sizeof(GLfloat), nullptr, GL_STATIC_DRAW);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, lutBuff);
    glDispatchCompute(width, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    lutProg.popProgram();

    GLfloat* data = (GLfloat*) glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    for(int i=0; i < lutImage.dataSize(); ++i)
        lutImage.pixels()[i] = (unsigned char)(data[i] * 255.0);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glDeleteBuffers(1, &lutBuff);

    const string IMAGE_NAME = "QualityGradient";
    getImageBank().addImage(IMAGE_NAME, lutImage);

    return IMAGE_NAME;
}
