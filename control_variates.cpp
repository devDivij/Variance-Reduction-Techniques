#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
using namespace std;

double gen_unif(mt19937 &gen)
{
    uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}
double calculateVariance(double mean, const vector<double> &data)
{

    double accum = 0.0;

    for (double d : data)
    {
        accum += (d - mean) * (d - mean);
    }
    return accum / (data.size() - 1);
}
double calculateCovariance(double mean1, double mean2, const vector<double> &v1, const std::vector<double> &v2)
{
    if (v1.size() <= 1)
        return 0.0;
    double accum = 0.0;

    for (size_t i = 0; i < v1.size(); ++i)
    {
        accum += (v1[i] - mean1) * (v2[i] - mean2);
    }

    return accum / (v1.size() - 1);
}
int main()
{
    int n = 100000;
    vector<double> Y, X;
    random_device rd;
    mt19937 gen(rd());
    for (int i = 0; i < n; ++i)
    {
        double u = gen_unif(gen);
        Y.push_back(exp(u * u));
        X.push_back(u * u);
    }
    double E1r = 1.0 / 3;
    double E1o = accumulate(X.begin(), X.end(), 0.0) / n;
    double E2o = accumulate(Y.begin(), Y.end(), 0.0) / n;
    double b = calculateCovariance(E1o, E2o, X, Y) / calculateVariance(E1o, X);
    cout << fixed << setprecision(8);
    cout << "Variance" << "\t" << "Mean" << endl;
    cout << "------------------------------------" << endl;
    cout << calculateVariance(E2o, Y) << '\t' << E2o << " (Original)" << endl;
    for (int i = 0; i < n; ++i)
    {
        Y[i] = Y[i] - b * (X[i] - E1r);
    }
    double E2r = E2o - b * (E1o - E1r);
    cout << calculateVariance(E2r, Y) << '\t' << E2r << " (Adjusted)";
}