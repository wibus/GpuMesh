#version 440

in vec3 pos;
in vec3 col;

layout(location = 0) out vec4 FragColor;

void main(void)
{
    FragColor = vec4(col, 1);
}

