#version 440

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

in vec4 position;
in vec4 colour_in;

out vec4 mv_pos;
out vec4 colour_out;

void main()
{
    mv_pos = view * model * position;
    gl_Position = proj * mv_pos;
    colour_out = colour_in;
}