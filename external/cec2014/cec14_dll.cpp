double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
void cec14_test_func(double *, double *, int, int, int);
extern "C" __declspec(dllexport) void cec14_eval(double *x, double *f, int nx, int mx, int func_num)
{
    cec14_test_func(x, f, nx, mx, func_num);
}
