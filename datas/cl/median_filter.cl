void sort(float *a, float *b, float *c) {
   int swap;
   if(*a > *b) {
      swap = *a;
      *a = *b;
      *b = swap;
   }
   if(*a > *c) {
      swap = *a;
      *a = *c;
      *c = swap;
   }
   if(*b > *c) {
      swap = *b;
      *b = *c;
      *c = swap;
   }
}
__kernel void median_filter(__global float *img, __global float *result, int width,int height)
{
    int w = width;
    int h = height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;
    // Keeping the edge pixels the same
    if( posx == 0 || posy == 0 || posx == w-1 || posy == h-1 )
    {
        result[i] = img[i];
    }
    else
    {
        float pixel00, pixel01, pixel02, pixel10, pixel11, pixel12, pixel20, pixel21, pixel22;
        pixel00 = img[i - 1 - w];
        pixel01 = img[i- w];
        pixel02 = img[i + 1 - w];
        pixel10 = img[i - 1];
        pixel11 = img[i];
        pixel12 = img[i + 1];
        pixel20 = img[i - 1 + w];
        pixel21 = img[i + w];
        pixel22 = img[i + 1 + w];
        //sort the rows
        sort( &(pixel00), &(pixel01), &(pixel02) );
        sort( &(pixel10), &(pixel11), &(pixel12) );
        sort( &(pixel20), &(pixel21), &(pixel22) );
        //sort the columns
        sort( &(pixel00), &(pixel10), &(pixel20) );
        sort( &(pixel01), &(pixel11), &(pixel21) );
        sort( &(pixel02), &(pixel12), &(pixel22) );
        //sort the diagonal
        sort( &(pixel00), &(pixel11), &(pixel22) );
        // median is the the middle value of the diagonal
        result[i] = pixel11;
    }
}