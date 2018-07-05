/* Please Write the OpenCL Kernel(s) code here*/
float2 hex_coord(float2 coord){
    float H = 0.01f;
    float S = ((3.0f / 2.0f) * H / sqrt(3.0f));
    
    int i = (int)(coord.x);
    int j = (int)(coord.y);
    
    float2 r = (float2)(0.0f,0.0f);
    r.x = (float)i * S;
    r.y = (float)i * H + (float)(i % 2) * H / 2.0f;
    
    return r;
}

float2 hex_index(float2 coord){
    float H = 0.01f;
    float S = ((3.0f / 2.0f) * H / sqrt(3.0f));
    
    float2 r;
    float x = coord.x;
    float y = coord.y;
    
    int it = (int)(floor(x / S));
    float yts = y - (float)(it % 2) * H / 2.0f;
    int jt = (int)(floor((float)it) * S);
    float xt = x - (float)it * S;
    float yt = yts - (float)jt * H;
    
    int deltaj = (yt > H / 2.0f) ? 1 : 0;
    float fcond = S * (2.0f / 3.0f) * fabs(0.5f - yt / H);
    
    if(xt > fcond){
        r.x = it;
        r.y = jt;
    }else{
        r.x = it - 1;
        r.y = jt - fmod(r.x, 2.0f) + deltaj;
    }
    
    return r;
}

__kernel void hexpix_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 hexIx = hex_index(coord);
   float2 hexXy = hex_coord(hexIx);
   float4 fcol = read_imagef(input,sampler,hexXy);
   
   write_imagef(output,convert_int2(coord),fcol);
   
}