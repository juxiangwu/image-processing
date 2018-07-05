/* Please Write the OpenCL Kernel(s) code here*/

void color_matrix_4x5_internal(__read_only image2d_t input,__write_only image2d_t output,float * mask){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord) * 255.0f;
   
   float4 rgba;
  
   rgba.x = mask[0] * color.x + mask[1] * color.y + mask[2] * color.z + mask[3] * color.w + mask[4];
   rgba.y = mask[0 + 5] * color.x + mask[1 + 5] * color.y + mask[2 + 5] * color.z + mask[3 + 5] * color.w + mask[4 + 5];
   rgba.z = mask[0 + 5 * 2] * color.x + mask[1 + 5 * 2] * color.y + mask[2 + 5 * 2] * color.z + mask[3 + 5 * 2] * color.w + mask[4 + 5 * 2];
   rgba.w = mask[0 + 5 * 3] * color.x + mask[1 + 5 * 3] * color.y + mask[2 + 5 * 3] * color.z + mask[4 + 5 * 3] * color.w + mask[4 + 5 * 3];
   
   rgba = clamp(rgba,0.0f,255.0f);
   
   rgba /= 255.0f;
   
   write_imagef(output,convert_int2(coord),rgba);
}
__kernel void polariod_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    float color_matrix[20] = {
           1.438f,-0.062f,-0.062f,0,0,
			-0.122f,1.378f,-0.122f,0,0,
			-0.016f,-0.016f,1.483f,0,0,
			0,0,0,1,0
    }; 
  
    color_matrix_4x5_internal(input,output,color_matrix);
}