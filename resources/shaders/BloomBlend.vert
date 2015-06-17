#version 440

layout(location=0) in vec2 position;

out vec2 texCoord;


void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    texCoord = (position + 1.0) / 2.0;
}
