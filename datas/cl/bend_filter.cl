/* Please Write the OpenCL Kernel(s) code here*/

__kernel void bend_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   
   
   float height = (float)dim.y - coord.y;
   float offset = pow(height,2.5f);
   float u_time = 1.0f;
   float speed = 2.0f;
   float bendFactor = 0.2f;
   offset *= (sin(u_time * speed) * bendFactor);
   
   float4 color = read_imagef(input,sampler,(float2)(coord.x,coord.y + offset));
   
   write_imagef(output,convert_int2(coord),color);
}