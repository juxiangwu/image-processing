/* Please Write the OpenCL Kernel(s) code here*/
__kernel void worry_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   const float speed = 1.0f;
   const float bendFactor = 0.2f;
   const float timeAcceleration = 15.0f;
   const float utime = 1000.0f;
   const float waveRadius = 5.0f;
   
   float stepVal = (utime * timeAcceleration) + coord.x * 61.8f;
   float offset = cos(stepVal) * waveRadius;
   float2 iptr = (float2)(0.0f,0.0f);
   
   float4 color = read_imagef(input,sampler,(float2)(coord.x,coord.y + offset));
   
   write_imagef(output,convert_int2(coord),color);
}