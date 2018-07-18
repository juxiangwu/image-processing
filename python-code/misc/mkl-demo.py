from ctypes import *

# Load the share library
mkl = cdll.LoadLibrary("C:/Users/jenson/Anaconda3/Library/bin/mkl_rt.dll")
# For Intel MKL prior to version 10.3 us the created .so as below
# mkl = dll.LoadLibrary("./libmkl4py.so")
cblas_dgemm = mkl.cblas_dgemm


def print_mat(mat, m, n):
  for i in range(0,m):
    print (" "),
    for j in range(0,n):
      print (mat[i*n+j])
    print('')

# Initialize scalar data
Order = 101  # 101 for row-major, 102 for column major data structures
TransA = 111 # 111 for no transpose, 112 for transpose, and 113 for conjugate transpose
TransB = 111
m = 2
n = 4
k = 3
lda = k
ldb = n
ldc = n
alpha = 1.0
beta = -1.0

# Create contiguous space for the double precision array
amat = c_double * 6      
bmat = c_double * 12
cmat = c_double * 8

# Initialize the data arrays
a = amat(1,2,3, 4,5,6)
b = bmat(0,1,0,1, 1,0,0,1, 1,0,1,0)
c = cmat(5,1,3,3, 11,4,6,9)

print("nMatrix A =")
print_mat(a,2,3) 
print("nMatrix B =")
print_mat(b,3,4)
print("nMatrix C =")
print_mat(c,2,4)

print("nCompute", alpha, "* A * B + ", beta, "* C")

# Call Intel MKL by casting scalar parameters and passing arrays by reference
cblas_dgemm( c_int(Order), c_int(TransA), c_int(TransB), 
             c_int(m), c_int(n), c_int(k), c_double(alpha), byref(a), c_int(lda), 
             byref(b), c_int(ldb), c_double(beta), byref(c), c_int(ldc))

print_mat(c,2,4)
print('')