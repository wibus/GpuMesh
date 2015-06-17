#version 440

uniform sampler2D Base;
uniform sampler2D Filter;

in vec2 texCoord;

layout(location=0) out vec4 FragColor;


void main()
{
    vec3 base = texture(Base, texCoord).rgb;
    float grain = texture(Filter, texCoord).a;
    FragColor = vec4(base + grain - 0.5, 1.0);
}
