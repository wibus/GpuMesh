#version 440

uniform sampler2D Filter;

in vec2 texCoord;

layout(location=0) out vec4 FragColor;


void main()
{
    FragColor = vec4(vec3(texture(Filter, texCoord).r), 1.0);
}
