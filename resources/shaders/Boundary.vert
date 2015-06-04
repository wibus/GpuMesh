#version 440

uniform mat4 Vmat;
uniform mat4 Pmat;
uniform vec3 Color;
uniform vec3 Offset;

layout(location=0) in vec3 position;

out vec3 pos;
out vec3 col;

void main(void)
{
    vec4 pos4 = Vmat * vec4(position, 1);
    gl_Position = Pmat * (pos4 + vec4(Offset, 0));
    pos = vec3(pos4);
    col = Color;
}

