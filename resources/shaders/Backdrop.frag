#version 440

uniform sampler2D Backdrop;

in vec2 texCoord;

layout(location=0) out vec4 FragColor;


void main()
{
    FragColor = texture(Backdrop, texCoord);
}
