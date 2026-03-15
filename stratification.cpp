#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

using namespace std;

double inverseNormalCDF(double p)
{

    // Acklacm Algorithm

    if (p <= 0.0 || p >= 1.0)
        return 0.0;

    static const double a[] = {
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00};

    static const double b[] = {
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01};

    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00};

    static const double d[] = {
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};

    const double p_low = 0.02425;
    const double p_high = 1.0 - p_low;

    double z;

    if (p < p_low)
    {
        double q = std::sqrt(-2.0 * std::log(p));
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    else if (p > p_high)
    {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    else
    {
        double q = p - 0.5;
        double r = q * q;
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }

    return z;
}
const double S0 = 100.0;
const double K = 105.0;
const double r = 0.05;
const double sigma = 0.2;
const double T = 1.0;
const int n = 100000;

double callPayoff(double S, double K)
{
    return max(S - K, 0.0);
}

void runStandardMonteCarlo(mt19937 &gen, normal_distribution<double> &dist)
{
    double sumPayoff = 0.0;
    double sumSqPayoff = 0.0;
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * sqrt(T);

    for (int i = 0; i < n; ++i)
    {
        double Z = dist(gen);
        double ST = S0 * exp(drift + vol * Z);
        double payoff = callPayoff(ST, K);

        sumPayoff += payoff;
        sumSqPayoff += payoff * payoff;
    }

    double meanPrice = exp(-r * T) * (sumPayoff / n);
    double variancePayoff = (sumSqPayoff / n) - (sumPayoff / n) * (sumPayoff / n);
    double varianceEstimator = variancePayoff / n;
    double stdError = sqrt(varianceEstimator) * exp(-r * T);

    cout << "--- Standard Monte Carlo ---" << endl;
    cout << "Estimated Price: " << meanPrice << endl;
    cout << "Std Error: " << stdError << endl;
}
void runStratifiedMonteCarlo(mt19937 &gen, uniform_real_distribution<double> &udist)
{
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * sqrt(T);
    int strata = 100;
    int n_eff = n / strata;

    double overallSum = 0.0;
    double varianceSumWithin = 0.0;

    for (int i = 0; i < strata; ++i)
    {
        double stratumSum = 0.0;
        double stratumSumSq = 0.0;

        for (int j = 0; j < n_eff; ++j)
        {
            double U = udist(gen);
            double Z = inverseNormalCDF((i + U) / static_cast<double>(strata));
            double ST = S0 * exp(drift + vol * Z);
            double payoff = callPayoff(ST, K);

            stratumSum += payoff;
            stratumSumSq += payoff * payoff;
        }

        double stratumMean = stratumSum / n_eff;
        double stratumVar = (stratumSumSq / n_eff) - stratumMean * stratumMean;

        overallSum += stratumMean;
        varianceSumWithin += stratumVar / n_eff;
    }

    double meanPayoff = overallSum / strata;
    double meanPrice = exp(-r * T) * meanPayoff;

    double varianceEstimator = varianceSumWithin / (strata * strata);
    double stdError = sqrt(varianceEstimator) * exp(-r * T);

    cout << "--- Stratified Monte Carlo ---" << endl;
    cout << "Estimated Price: " << meanPrice << endl;
    cout << "Std Error: " << stdError << endl;
}

int main()
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);
    uniform_real_distribution<double> udist(0.0, 1.0);

    cout << fixed << setprecision(5);
    cout << "Simulations: " << n << endl
         << endl;

    runStandardMonteCarlo(gen, dist);
    runStratifiedMonteCarlo(gen, udist);
    return 0;
}