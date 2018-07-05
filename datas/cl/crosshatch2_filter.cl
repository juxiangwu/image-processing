/* Please Write the OpenCL Kernel(s) code here*/
float4 postfx(__read_only image2d_t input,__read_only sampler_t sampler,float2 uv,float dim){
    float stitching_size = 6.0f;
    int invert = 0;
    
    float4 c = (float4)(0.0f,0.0f,0.0f,0.0f);
    float size = stitching_size;
    
    float2 cPos = uv * (float2)(dim,dim);
    float2 tlPos = floor(cPos / (float2)(size,size));
    
    tlPos *= size;
    
    int remX = (int)(fmod(cPos.x,size));
    int remY = (int)(fmod(cPos.y,size));
    
    if(remX == 0 && remY == 0){
        tlPos = cPos;
    }
    
    float2 blPos = tlPos;
    blPos.y += (size - 1.0f);
    
    if((remX == remY) || (((int)cPos.x - (int)blPos.x) == ((int)blPos.y - (int)cPos.y))){
        if(invert == 1){
            c = (float4)(0.2f,0.15f,0.05f,1.0f);
        }else{
            c = read_imagef(input,sampler,tlPos * (float2)(1.0f / dim,1.0f / dim)) * 1.4f;
        }
    }else{
        if(invert == 1){
             c = read_imagef(input,sampler,tlPos * (float2)(1.0f / dim,1.0f / dim)) * 1.4f;
        }else{
         c = (float4)(0.0f,0.0f,0.0f,1.0f);
        }
    }
    
    return c;
}

__kernel void crosshatch2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float dim = 600.0f;
   
   write_imagef(output,convert_int2(coord),postfx(input,sampler,coord,dim));
   
}