#version 430 core

uniform sampler2D texture;

in VS_OUT
{
	vec2 texc;
} fs_in;

out vec4 color;

uniform sampler2D color0; // The blurred image
uniform sampler2D scene;
uniform sampler2DShadow depth;

vec2 coefficients = vec2(0.6,0.6);

void main () {
    texCoord = fs_in.texc;
    vec4 blur = texture2D(texture, texCoord);
    vec4 orig = texture2D(texture, texCoord);

    gl_FragColor = coefficients.x * orig + coefficients.y * blur;

    gl_FragDepth = shadow2D(depth, vec3(texCoord, 0.0)).x;
}