# include <iostream>
# include <cmath>
# include <fstream>

double f(double x){
    return x+pow(sin(x),2);
}

double f_prime(double x){
    return 1+ 2*sin(x)*cos(x);
}


int main(int argc, char** argv){
    double a = -1;
    double b = 1; 

    int num_points = 100;

    double h = double(b-a)/double(num_points-1); // Corrected step size

    double* X = new double[num_points];

    X[0] = a;
    X[num_points-1] = b; // Corrected boundary

    for(int i=1; i<num_points-1; i++){
        X[i]= a + i*h;
    }
    
    double* df = new double[num_points];

    for (int i = 0; i<num_points; i++){
        if(i == 0|| i==1){
            df[i] = (-f(X[i+2]) + 4*f(X[i+1]) - 3*f(X[i])) / (2*h);  // Higher-order forward difference
        }
        else if (i == num_points-1|| i== num_points-2){
            df[i] = (3*f(X[i]) - 4*f(X[i-1]) + f(X[i-2])) / (2*h);  // Higher-order backward difference
        }
        else {
            df[i] = (-f(X[i+2]) + 8*f(X[i+1]) - 8*f(X[i-1]) + f(X[i-2])) / (12*h);  // 5-point central difference
        }
    }

    std::ofstream fptr;
    fptr.open("10plot_data.txt");
    for (int i=0; i<num_points; i++){
        fptr << X[i] << " " << f_prime(X[i]) << " " << df[i] << std::endl;
    }
    fptr.close();


    delete[] df;
    delete[] X;
}
