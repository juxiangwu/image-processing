/* Please Write the OpenCL Kernel(s) code here*/

__kernel void luminace_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float4 color;

    float4 srcColor = read_imagef(input,sampler,coord);

    float gray = srcColor.x * 0.299f + srcColor.y * 0.587f + srcColor.z * 0.114f;
    
    int use_thresh = 1;
    if(use_thresh){
        float thresh = 0.32f;
  
         if(gray > thresh){
            color = (float4)(mix(srcColor,(float4)(1.0f),0.5f).xyz,1.0f);
        }else{
            color = (float4)((float3)(0.0f),1.0f);    
        }
         write_imagef(output,coord,color);
    }else{
         write_imagef(output,coord,(float4)(gray,gray,gray,1.0f));
    }
   
}
