#version 430 core

uniform sampler2D texture;
uniform sampler2D texture_map;

in VS_OUT
{
	vec2 texc;
} fs_in;

out vec4 color;

void main(void)
{
    vec3 rgb = texture2D(texture, fs_in.texc).rgb;
    // vec3 rgb2 = texture2D(texture, fs_in.texc).rgb;
    rgb = vec3(texture2D(texture_map, vec2(rgb.r, .16666)).r,
                 texture2D(texture_map, vec2(rgb.g, .5)).g,
                 texture2D(texture_map, vec2(rgb.b, .83333)).b);
    color = vec4(rgb,1.0);
}