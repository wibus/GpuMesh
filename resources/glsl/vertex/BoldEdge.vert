#version 440

uniform mat4 ProjMat;
uniform mat4 ViewMat;
uniform vec4 CutPlaneEq;


layout(location = 0) in vec3 position;


out float dist;


void main()
{
    dist = dot(position, CutPlaneEq.xyz) - CutPlaneEq.w;
    vec4 viewPos = (ViewMat * vec4(position, 1.0));
    const float OFFSET = 0.99975; // Distance relative offset
    gl_Position = ProjMat * vec4(viewPos.xy, viewPos.z * OFFSET, viewPos.w);
}
