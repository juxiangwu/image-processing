/* Please Write the OpenCL Kernel(s) code here*/

__kernel void shi_tomasi_feather_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                                  
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float derivativeDifference = color.x - color.y;
   float zElement = (color.z * 2.0f) - 1.0f;
   
   float cornerness = color.x + color.y - sqrt(derivativeDifference * derivativeDifference + 4.0f * zElement * zElement);
   float sensitivity = 1.5f;
   float rgba = cornerness * sensitivity;
   
   write_imagef(output,convert_int2(coord),(float4)(rgba,rgba,rgba,1.0f));
}