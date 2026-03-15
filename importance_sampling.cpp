#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
using namespace std;

const double S0 = 100.0;
const double K = 150.0;
const double T = 1.0;
const double r = 0.05;
const double sigma = 0.2;
const int n = 10000;

double calculate_ST(double z)
{
    return S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * z);
}

double payoff(double ST)
{
    return max(ST - K, 0.0);
}

void runStandardMonteCarlo(mt19937 &gen, normal_distribution<double> &dist)
{
    double sumPayoff = 0.0;
    double sumSqPayoff = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double z = dist(gen);
        double ST = calculate_ST(z);
        double p = payoff(ST);

        sumPayoff += p;
        sumSqPayoff += p * p;
    }

    double meanPayoff = sumPayoff / n;
    double meanPrice = exp(-r * T) * meanPayoff;
    double variancePayoff = (sumSqPayoff / n) - meanPayoff * meanPayoff;
    double varianceEstimator = variancePayoff / n;
    double stdError = sqrt(varianceEstimator) * exp(-r * T);

    cout << "--- Standard Monte Carlo ---" << endl;
    cout << "Estimated Price: " << meanPrice << endl;
    cout << "Std Error: " << stdError << endl;
}

void runImportanceSampling(mt19937 &gen, normal_distribution<double> &dist)
{
    double h = (log(K / S0) - (r - 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double sumPayoff = 0.0;
    double sumSqPayoff = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double z = dist(gen);
        double z_shifted = z + h;
        double ST = calculate_ST(z_shifted);
        double p = payoff(ST);
        double weight = exp(-h * z - 0.5 * h * h);
        double weighted_payoff = p * weight;

        sumPayoff += weighted_payoff;
        sumSqPayoff += weighted_payoff * weighted_payoff;
    }

    double meanPayoff = sumPayoff / n;
    double meanPrice = exp(-r * T) * meanPayoff;
    double variancePayoff = (sumSqPayoff / n) - meanPayoff * meanPayoff;
    double varianceEstimator = variancePayoff / n;
    double stdError = sqrt(varianceEstimator) * exp(-r * T);

    cout << "--- Importance Sampling ---" << endl;
    cout << "Estimated Price: " << meanPrice << endl;
    cout << "Std Error: " << stdError << endl;
}

int main()
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);

    cout << fixed << setprecision(5);
    cout << "Total Simulations: " << n << endl
         << endl;

    runStandardMonteCarlo(gen, dist);
    cout << endl;
    runImportanceSampling(gen, dist);

    return 0;
}