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

int main()
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);
    double sum_payoff = 0.0;
    double sum_sq_payoff = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double z = dist(gen);
        double ST = calculate_ST(z);
        double p = payoff(ST);
        sum_payoff += p;
        sum_sq_payoff += p * p;
    }

    double price_std = exp(-r * T) * (sum_payoff / n);
    double var_std = (sum_sq_payoff / n) - (sum_payoff / n) * (sum_payoff / n);

    double h = log(K / S0) / (sigma * sqrt(T));

    double sum_payoff_is = 0.0;
    double sum_sq_payoff_is = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double z = dist(gen);

        double z_shifted = z + h;

        double ST = calculate_ST(z_shifted);

        double p = payoff(ST);
        double weight = exp(-h * z_shifted + 0.5 * h * h);

        double weighted_payoff = p * weight;

        sum_payoff_is += weighted_payoff;
        sum_sq_payoff_is += weighted_payoff * weighted_payoff;
    }

    double price_is = exp(-r * T) * (sum_payoff_is / n);
    double var_is = (sum_sq_payoff_is / n) - (sum_payoff_is / n) * (sum_payoff_is / n);

    cout << fixed << setprecision(5);
    cout << "--- Standard MC ---" << endl;
    cout << "Price: " << price_std << endl;
    cout << "Variance: " << var_std << endl;

    cout << "--- Importance Sampling ---" << endl;
    cout << "Price: " << price_is << endl;
    cout << "Std Error: " << var_is << endl;

    return 0;
}