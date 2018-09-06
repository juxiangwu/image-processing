#include <iostream>
#include <dlib/matrix.h>


int main()
{
    // declare a matrix
    dlib::matrix<double, 3, 1> y;
    dlib::matrix<double, 3, 3> M;
    // initialize matrices
    M = 54.2, 65.2, 43,
        23.4, 12.3, 55.4,
        11, 34.6, 78.9;
    y = 3.5,
        1.2,
        6.7;
    // solve this linear system
    dlib::matrix<double> x = dlib::inv(M)*y;
    std::cout << "x: \n" << x << "\n";
    // check if the calculation is correct
    std::cout << "Check: M*x- y= \n" << M*x - y << "\n";

    // sum the matrix
    double sumM = 0;
    for (int r = 0; r < M.nr(); ++r)
    {
        for (int c = 0; c < M.nc(); ++c)
        {
            sumM += M(r, c);
        }
    }
    std::cout << "Sum of matrix by looping= " << sumM << "\n";
    std::cout << "Sum of matrix by sum() function= " << dlib::sum(M) << "\n";
    // print using comma separator
    std::cout <<"print matrix M using comma delimiter: \n" <<dlib::csv << M << "\n";

    std::system("pause");
    return 0;
}
