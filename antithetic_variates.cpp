#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
using namespace std;

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

    double meanPayoff = sumPayoff / n;
    double meanPrice = exp(-r * T) * meanPayoff;
    double variancePayoff = (sumSqPayoff / n) - meanPayoff * meanPayoff;
    double varianceEstimator = variancePayoff / n;
    double stdError = sqrt(varianceEstimator) * exp(-r * T);

    cout << "--- Standard Monte Carlo ---" << endl;
    cout << "Estimated Price: " << meanPrice << endl;
    cout << "Std Error: " << stdError << endl;
}

void runAntitheticMonteCarlo(mt19937 &gen, normal_distribution<double> &dist)
{
    double sumPayoff = 0.0;
    double sumSqPayoff = 0.0;
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * sqrt(T);

    for (int i = 0; i < n / 2; ++i)
    {
        double Z = dist(gen);

        double ST1 = S0 * exp(drift + vol * Z);
        double payoff1 = callPayoff(ST1, K);

        double ST2 = S0 * exp(drift + vol * (-Z));
        double payoff2 = callPayoff(ST2, K);

        double pairAverage = 0.5 * (payoff1 + payoff2);

        sumPayoff += pairAverage;
        sumSqPayoff += pairAverage * pairAverage;
    }

    int n_pairs = n / 2;
    double meanPayoff = sumPayoff / n_pairs;
    double meanPrice = exp(-r * T) * meanPayoff;
    double variancePayoff = (sumSqPayoff / n_pairs) - meanPayoff * meanPayoff;
    double varianceEstimator = variancePayoff / n_pairs;
    double stdError = sqrt(varianceEstimator) * exp(-r * T);

    cout << "--- Antithetic Variates ---" << endl;
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
    runAntitheticMonteCarlo(gen, dist);

    return 0;
}