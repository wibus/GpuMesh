#version 440

uniform vec2 TexScale;

layout(location=0) in vec2 position;

out vec2 texCoord;
out vec2 baseCoord;


void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    texCoord = (position * TexScale + 1.0) / 2.0;
    baseCoord = (position + 1.0) / 2.0;
}
