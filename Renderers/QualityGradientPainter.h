#ifndef GPUMESH_QUALITYGRADIENTPAINTER
#define GPUMESH_QUALITYGRADIENTPAINTER

#include <string>


class QualityGradientPainter
{
public:
    QualityGradientPainter();
    ~QualityGradientPainter();

    virtual std::string generate(int width, int height, int minHeight);
};

#endif // GPUMESH_QUALITYGRADIENTPAINTER
