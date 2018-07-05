/* Please Write the OpenCL Kernel(s) code here*/
__kernel void yuv2rgb_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                             
  
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
  
    float4 protanopia_mask_1 = (float4)(1,1,1,0);
    float4 protanopia_mask_2 = (float4)(0,-0.187f,1.8556f,0);
    float4 protanopia_mask_3 = (float4)(1.5701f,-0.4664f,0,0);
    float4 protanopia_mask_4 = (float4)(0.0f,0.0f,0.0f,1.0f);
    
    float4 protanopia_mask[4] = {protanopia_mask_1,protanopia_mask_2,protanopia_mask_3,protanopia_mask_4};
   
    //float4 v1 = protanopia_mask[0];
    float4 srcRGBA = read_imagef(input,sampler,coord);
    
    float4 dstRGBA;
    float sum[4] = {0.0f};
    for(int i = 0;i < 4;i++){
        
        sum[i] += dot(protanopia_mask[i],srcRGBA);
    }
    
    dstRGBA = (float4)(sum[0],sum[1],sum[2],sum[3]);
    
    write_imagef(output,coord,dstRGBA);
}