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
__kernel void kodachrome_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    float color_matrix[20] = {
           1.1285582396593525f,-0.3967382283601348f,-0.03992559172921793f,0,63.72958762196502f,
			-0.16404339962244616f,1.0835251566291304f,-0.05498805115633132f,0,24.732407896706203f,
			-0.16786010706155763f,-0.5603416277695248f,1.6014850761964943f,0,35.62982807460946f,
			0,0,0,1,0
    }; 
   
   color_matrix_4x5_internal(input,output,color_matrix);
}