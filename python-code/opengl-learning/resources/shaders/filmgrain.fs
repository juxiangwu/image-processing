#version 430 core

uniform sampler2D texture;

in VS_OUT
{
	vec2 texc;
} fs_in;

out vec4 color;
//noise effect intensity value (0 = no effect, 1 = full effect)
const float noiseIntensity = 0.2;
//scanlines effect intensity value (0 = no effect, 1 = full effect)
const float scanlineIntensity = 0.1;
//scanlines effect count value (0 = no effect, 4096 = full effect)
const float scanlineCount = 2048.0;

float time = 0.8;
void main() {
    vec2 texCoord = fs_in.texc;
    vec4 rgba = texture2D(texture, texCoord).rgba;
    
    // make some noise
	float x = texCoord.x * texCoord.y * time;
	x = mod(x, 13.0) * mod(x, 123.0);
	float dx = mod(x, 0.006);
	
	// add noise
	vec3 res = rgba.rgb + rgba.rgb * clamp(0.1 + dx * 100.0, 0.0, 1.0);

	// get us a sine and cosine
	//vec2 sc = vec2(sin(texCoord.y * scanlineCount),cos(texCoord.y * scanlineCount));

	// add scanlines
	//res += color.rgb * vec3(sc.x, sc.y, sc.x) * scanlineIntensity;
	
	// interpolate between source and result by intensity
	res = mix(rgba.rgb, res, noiseIntensity);

	// return with source alpha
	color =  vec4(res, 1.0);
    
}