/* Please Write the OpenCL Kernel(s) code here*/

__kernel void mosaic2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   
   float t = 1.2f;
   float2 position = (float2)(coord.x / dim.x,1.0f - coord.y / dim.y);
   float2 samplePos = position.xy;
   float pixel = 64.0f;
   float edges = 0.02f;
   float depth = 8.0f;
   float shift = 5.0f;
   
   samplePos.x = floor(samplePos.x * (dim.x / pixel)) / (dim.x / pixel);
   samplePos.y = floor(samplePos.y * (dim.y / pixel)) / (dim.y / pixel);
   
   float st = sin(t * 0.05f);
   float ct = cos(t * 0.05f);
   
   float h = st * shift / dim.x;
   float v = ct * shift / dim.y;
   
   float3 o = read_imagef(input,sampler,samplePos).xyz;
   float r = read_imagef(input,sampler,samplePos +  (float2)(+h,+v)).x;
   float g = read_imagef(input,sampler,samplePos +  (float2)(-h,-v)).y;
   float b = read_imagef(input,sampler,samplePos).z;
   float iptr;
   r = mix(o.x,r,fract(fabs(st),&iptr));
   g = mix(o.y,g,fract(fabs(ct),&iptr));
   
   float n = fmod(coord.x,pixel) * edges;
   float m = fmod(dim.y - coord.y,pixel) * edges;
   
   float3 c = (float3)(r,g,b);
   
   c = floor(c * depth) / depth;
   c = c * (1.0f - (m + n) * (n + m));
   write_imagef(output,convert_int2(coord),(float4)(c,1.0f));

}