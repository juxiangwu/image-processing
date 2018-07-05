/* Please Write the OpenCL Kernel(s) code here*/
void color_matrix_4x5_internal(__read_only image2d_t input,__write_only image2d_t output,float * mask);
__kernel void desaturation_filter(__read_only image2d_t input,__write_only image2d_t output);
__kernel void desaturation_filter(__read_only image2d_t input,__write_only image2d_t output){
   
   float amount = -1.0f;
   float x = amount * 2.0f / 3.0f + 1.0f;
   float y = (x - 1.0f) * -0.5f;
   float color_matrix [] = {
       x,y,y,0,0,
       y,x,y,0,0,
       y,y,x,0,0,
       0,0,0,1,0
   };
   color_matrix_4x5_internal(input,output,color_matrix);
}

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
   /*
   rgba.x = mask[0] * color.x + mask[1] * color.y + mask[2] * color.z + mask[4];
   rgba.y = mask[5] * color.x + mask[6] * color.y + mask[7] * color.z + mask[9];
   rgba.z = mask[10] * color.x + mask[11] * color.y + mask[12] * color.z + mask[14];
   rgba.w = color.w;
   */
   if(rgba.x < 0.0f){
       rgba.x = 0.0f;
   }
   
   if(rgba.x > 255.0f){
       rgba.x = 255.0f;
   }
   
    if(rgba.y < 0.0f){
       rgba.y = 0.0f;
   }
   
   if(rgba.y > 255.0f){
       rgba.y = 255.0f;
   }
   
    if(rgba.z < 0.0f){
       rgba.z = 0.0f;
   }
   
   if(rgba.z > 255.0f){
       rgba.z = 255.0f;
   }
   
    if(rgba.w < 0.0f){
       rgba.w = 0.0f;
   }
   
   if(rgba.w > 255.0f){
       rgba.w = 255.0f;
   }
   
   rgba /= 255.0f;
   
   write_imagef(output,convert_int2(coord),rgba);
}