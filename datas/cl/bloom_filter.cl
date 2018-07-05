/* Please Write the OpenCL Kernel(s) code here*/

__kernel void bloom_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    float2 coord = (float2)(get_global_id(0),get_global_id(1));
    
    
    float4 sum = (float4)(0.0f,0.0f,0.0f,0.0f);
    float a = 0.15f;
    float g = 0.15f;
    float e = 0.25f;
    float b = 0.25f;
    float f = 0.35f;
    float c = 0.55f;
    float d = 0.45f;
    
    sum += read_imagef(input,sampler,coord + (float2)(-3,-4) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,-3) * a) * g;
    
    sum += read_imagef(input,sampler,coord + (float2)(-4,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,-2) * a) * g;
    
    sum += read_imagef(input,sampler,coord + (float2)(-4,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,-1) * a) * g;
    
    sum += read_imagef(input,sampler,coord + (float2)(-4,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,0) * a) * g;
    
    sum += read_imagef(input,sampler,coord + (float2)(-4,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,1) * a) * g;

    sum += read_imagef(input,sampler,coord + (float2)(-4,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,2) * a) * g;
    
    float4 color = read_imagef(input,sampler,coord);
    
    if(color.x < e){
        float4 rgba = sum * sum * b + color;
        rgba.w = 1.0f;
        write_imagef(output,convert_int2(coord),rgba);
    }else{
        if(color.x < f){
            float4 rgba = sum * sum * c + color;
            rgba.w = 1.0f;
            write_imagef(output,convert_int2(coord),rgba);
        }else{
            float4 rgba = sum * sum * d + color;
            rgba.w = 1.0f;
            write_imagef(output,convert_int2(coord),rgba);
        }
    }
}