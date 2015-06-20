#version 440

uniform sampler2D Base;
uniform sampler2D Filter;

in vec2 texCoord;
in vec2 baseCoord;

layout(location=0) out vec4 FragColor;


void main()
{
    vec3 base = texture(Base, baseCoord).rgb;
    float brush = texture(Filter, texCoord).b;
    FragColor = vec4(base * (0.5 + brush), 1.0);
}
