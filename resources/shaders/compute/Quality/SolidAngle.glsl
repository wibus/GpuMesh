#version 440


struct Tet
{
    int v[4];
};

struct Pri
{
    int v[6];
};

struct Hex
{
    int v[8];
};


// Vertices positions
// vec4 is only for alignment
layout (std140, binding=0) buffer Vert
{
    vec4 vert[];
};


float solidAngle(in vec3 a, in vec3 b, in vec3 c)
{
    float al = length(a);
    float bl = length(b);
    float cl = length(c);

    return abs(determinant(mat3(a, b, c))) /
            sqrt( 2.0 * (al*bl + dot(a, b)) *
                        (bl*cl + dot(b, c)) *
                        (cl*al + dot(c, a)));
}

float tetQuality(Tet tet)
{
    vec3 t0 = vec3(vert[tet.v[0]]);
    vec3 t1 = vec3(vert[tet.v[1]]);
    vec3 t2 = vec3(vert[tet.v[2]]);
    vec3 t3 = vec3(vert[tet.v[3]]);

    float q0 = solidAngle(t1 - t0, t2 - t0, t3 - t0);
    float q1 = solidAngle(t0 - t1, t2 - t1, t3 - t1);
    float q2 = solidAngle(t0 - t2, t1 - t2, t3 - t2);
    float q3 = solidAngle(t0 - t3, t1 - t3, t2 - t3);

    float minQ = min(min(q0, q1), min(q2, q3));
    return minQ * 3.67423461417; // 9 / sqrt(6)
}

float priQuality(Pri pri)
{
    vec3 t0 = vec3(vert[pri.v[0]]);
    vec3 t1 = vec3(vert[pri.v[1]]);
    vec3 t2 = vec3(vert[pri.v[2]]);
    vec3 t3 = vec3(vert[pri.v[3]]);
    vec3 t4 = vec3(vert[pri.v[4]]);
    vec3 t5 = vec3(vert[pri.v[5]]);

    float q0 = solidAngle(t1 - t0, t2 - t0, t4 - t0);
    float q1 = solidAngle(t0 - t1, t3 - t1, t5 - t1);
    float q2 = solidAngle(t0 - t2, t3 - t2, t4 - t2);
    float q3 = solidAngle(t1 - t3, t2 - t3, t5 - t3);
    float q4 = solidAngle(t0 - t4, t2 - t4, t5 - t4);
    float q5 = solidAngle(t1 - t5, t3 - t5, t4 - t5);

    float minQ = min(min(q0, q1),
                     min(min(q2, q3),
                         min(q4, q5)));
    return minQ * 2.61312592975; // 1.0 / <max val for regular prism>
}

float hexQuality(Hex hex)
{
    vec3 t0 = vec3(vert[hex.v[0]]);
    vec3 t1 = vec3(vert[hex.v[1]]);
    vec3 t2 = vec3(vert[hex.v[2]]);
    vec3 t3 = vec3(vert[hex.v[3]]);
    vec3 t4 = vec3(vert[hex.v[4]]);
    vec3 t5 = vec3(vert[hex.v[5]]);
    vec3 t6 = vec3(vert[hex.v[6]]);
    vec3 t7 = vec3(vert[hex.v[7]]);

    float q0 = solidAngle(t1 - t0, t2 - t0, t4 - t0);
    float q1 = solidAngle(t0 - t1, t3 - t1, t5 - t1);
    float q2 = solidAngle(t0 - t2, t3 - t2, t6 - t2);
    float q3 = solidAngle(t1 - t3, t2 - t3, t7 - t3);
    float q4 = solidAngle(t0 - t4, t5 - t4, t6 - t4);
    float q5 = solidAngle(t1 - t5, t4 - t5, t7 - t5);
    float q6 = solidAngle(t2 - t6, t4 - t6, t7 - t6);
    float q7 = solidAngle(t3 - t7, t5 - t7, t6 - t7);

    float minQ = min(min(min(q0, q1), min(q2, q3)),
                     min(min(q4, q5), min(q6, q7)));
    return minQ * 1.41421356237; // sqrt(2)
}
