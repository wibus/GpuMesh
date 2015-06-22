#version 440

uniform sampler2D BloomBase;
uniform sampler2D BloomBlur;

in vec2 texCoord;

layout(location=0) out vec4 FragColor;


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
    return texel.rgb * kernel[kId];
}


void main()
{
    vec3 blur = conv(texture(BloomBlur, texCoord), 12);

    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -12)), 0);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -11)), 1);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -10)), 2);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -9)) , 3);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -8)) , 4);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -7)) , 5);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -6)) , 6);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -5)) , 7);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -4)) , 8);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -3)) , 9);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -2)) , 10);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, -1)) , 11);

    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 1))  , 13);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 2))  , 14);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 3))  , 15);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 4))  , 16);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 5))  , 17);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 6))  , 18);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 7))  , 19);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 8))  , 20);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 9))  , 21);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 10)) , 22);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 11)) , 23);
    blur += conv(textureOffset(BloomBlur, texCoord, ivec2(0, 12)) , 24);


    vec3 base = texture(BloomBase, texCoord).rgb;
    FragColor = vec4(min(base + blur, vec3(1 + 8.0/256.0)), 1.0);
}
