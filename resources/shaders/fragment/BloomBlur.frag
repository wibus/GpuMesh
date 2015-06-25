#version 440

uniform sampler2D BloomBase;

in vec2 texCoord;

layout(location=0) out vec4 BloomBlur;


const int kernelSize = 25;
const float kernel[] = {
    0.001107962102984944, 0.0022733906253986705,
    0.004382075123393884, 0.007934912958920022,
    0.013497741628302402, 0.021569329706636487,
    0.032379398916485856, 0.0456622713472737,
    0.06049268112980998, 0.07528435803873115,
    0.08801633169111, 0.09666702920075089,
    0.09973557010039798,
    0.09666702920075089, 0.08801633169111,
    0.07528435803873115, 0.06049268112980998,
    0.0456622713472737, 0.032379398916485856,
    0.021569329706636487, 0.013497741628302402,
    0.007934912958920022, 0.004382075123393884,
    0.0022733906253986705, 0.001107962102984944
};

vec3 conv(in vec4 texel, in int kId)
{
    vec3 lum = texel.rgb;
    float blurIntensity = length(lum);
    blurIntensity = max(blurIntensity - 0.75, 0) * 1.0;
    lum *= blurIntensity * blurIntensity * blurIntensity;

    return lum * kernel[kId];
}

void main()
{
    vec3 blur = conv(texture(BloomBase, texCoord), 12);

    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-12, 0)), 0);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-11, 0)), 1);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-10, 0)), 2);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-9, 0)) , 3);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-8, 0)) , 4);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-7, 0)) , 5);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-6, 0)) , 6);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-5, 0)) , 7);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-4, 0)) , 8);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-3, 0)) , 9);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-2, 0)) , 10);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(-1, 0)) , 11);

    blur += conv(textureOffset(BloomBase, texCoord, ivec2(1, 0))  , 13);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(2, 0))  , 14);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(3, 0))  , 15);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(4, 0))  , 16);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(5, 0))  , 17);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(6, 0))  , 18);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(7, 0))  , 19);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(8, 0))  , 20);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(9, 0))  , 21);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(10, 0)) , 22);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(11, 0)) , 23);
    blur += conv(textureOffset(BloomBase, texCoord, ivec2(12, 0)) , 24);


    BloomBlur = vec4(blur, 1.0);
}
