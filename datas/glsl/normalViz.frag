#version 440

in vec4 mv_pos;
in vec4 colour_out;

out vec4 fragColour;

void main()
{
    vec3 normal_out = normalize(cross(dFdx(mv_pos.xyz), dFdy(mv_pos.xyz)));
    fragColour = vec4(normal_out, 1.0);
}