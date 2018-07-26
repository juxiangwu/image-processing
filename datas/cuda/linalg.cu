// VECTOR OPS
__global__
void dot_product(Vector* v1, Vector* v2, float* dest)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < v1->rows * v1->stride)
    {
        atomicAdd(dest, v1->data[index] * v2->data[index]);
    }
}

__global__
void lp_norm(Vector* v, float* dest, int p)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < v->rows * v->stride)
    {
        atomicAdd(dest, powf(v->data[index], p));
    }
    __syncthreads();
    *dest = powf(*dest, 1.0 / p);
}

__global__
void vector_reflection(Vector* v, float axis, float* origin, Vector* dest)
{ }

__global__
void vector_rotation(Vector* v, float radians, float* origin, Vector* dest)
{ }

__global__
void vector_reduction(Vector* v, REDUCTION_TYPE type, void* dest)
{ }

// MATRIX OPS
__global__
void add_matrices(Matrix* m1, Matrix* m2, Matrix* dest)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (!dest) { dest = m1; }
    if (index < m1->rows * m1->stride)
    {
        dest->data[index] = m1->data[index] + m2->data[index];
    }
}

__global__
void scalar_multiply_matrix(Matrix* m, float s, Matrix* dest)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (!dest) { dest = m; }
    if (index < m->rows * m->stride)
    {
        dest->data[index] = m->data[index] * s;
    }
}


__global__
void multiply_matrices(Matrix* left, Matrix* right, Matrix* dest)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dest->rows * dest->stride)
    {
        int row = index / dest->stride;
        int col = index % dest->stride;
        for (int i = 0; i < left->cols; i++)
        {
            dest->data[index] += left->data[row * dest->stride + i] *
                right->data[i * dest->stride + col];
        }
    }
}
        
__global__
void transpose_matrix(Matrix* m, Matrix* dest)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dest->rows * dest->stride)
    {
        int row = index / m->stride;
        int col = index % m->stride;
        int s = row * m->stride + col;
        int d = col * dest->stride + row;
        dest->data[d] = m->data[s];
    }
}
        