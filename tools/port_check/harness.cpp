// Evaluates the official CEC 2017 function on points read from a binary file.
// usage: cec17.exe func nx mx points.bin values.bin   (run with input_data/ in the working directory)
#include <stdio.h>
#include <stdlib.h>
double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
void cec17_test_func(double *, double *, int, int, int);
int main(int argc, char **argv)
{
    if (argc != 6) { fprintf(stderr, "usage: func nx mx in out\n"); return 2; }
    int f = atoi(argv[1]), nx = atoi(argv[2]), mx = atoi(argv[3]);
    double *x = (double *)malloc(sizeof(double) * nx * mx), *v = (double *)malloc(sizeof(double) * mx);
    FILE *fi = fopen(argv[4], "rb");
    if (!fi || fread(x, sizeof(double), (size_t)nx * mx, fi) != (size_t)nx * mx) { fprintf(stderr, "read error\n"); return 3; }
    fclose(fi);
    cec17_test_func(x, v, nx, mx, f);
    FILE *fo = fopen(argv[5], "wb");
    fwrite(v, sizeof(double), mx, fo);
    fclose(fo);
    return 0;
}
