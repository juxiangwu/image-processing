/* Please Write the OpenCL Kernel(s) code here*/
__kernel void tansform_filter(__read_only image2d_t input,__write_only image2d_t output){
   
    const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;
                               
    const int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int xoffset = -64;
    int yoffset = 64;
    
    int x1 = coord.x - xoffset;
    int y1 = coord.y - yoffset;
    
    if(x1 >= 0 && x1 < size.x && y1 >= 0 && y1 < size.y){
        float4 color = read_imagef(input,sampler,(int2)(x1,y1));
        write_imagef(output,coord,color);
    }else{
        write_imagef(output,coord,(float4)(1,1,1,1));
    }
}