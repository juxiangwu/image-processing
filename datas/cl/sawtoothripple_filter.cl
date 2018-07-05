/* Please Write the OpenCL Kernel(s) code here*/

__kernel void sawtoothripple_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float xAmplitude = 5.0f;
   float yAmplitude = 5.0f;
   float xWavelength = 16.0f;
   float yWavelength = 16.0f;
   
   float nx = coord.x / yWavelength;
   float ny = coord.y / yWavelength;
   
   float fx = fmod(nx,1.0f);
   float fy = fmod(ny,1.0f);
   
   float4 color = read_imagef(input,sampler,(float2)(coord.x + xAmplitude * fx,coord.y + yAmplitude * fy));
   
   write_imagef(output,convert_int2(coord),color);
}