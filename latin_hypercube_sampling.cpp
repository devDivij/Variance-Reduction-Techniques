#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <numeric>

using namespace std;
double inverseNormalCDF(double p)
{
    // Acklam Algorithm
    if (p <= 0.0 || p >= 1.0)
        return 0.0;

    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};

    const double p_low = 0.02425;
    const double p_high = 1.0 - p_low;

    double z;
    if (p < p_low)
    {
        double q = std::sqrt(-2.0 * std::log(p));
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    else if (p > p_high)
    {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    else
    {
        double q = p - 0.5;
        double r = q * q;
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    return z;
}

double callPayoff(double S, double K)
{
    return max(S - K, 0.0);
}

const double S0 = 100.0;
const double T = 1.0;
const double K = 105.0;
const double r = 0.05;
const double sigma = 0.2;
int m = 32;        // Time steps per path
int k = 10;        // Paths per batch
int n = 100000;    // Total paths
int n_eff = n / k; // Total batches
double dt = T / m;

void runLHSMonteCarlo(mt19937 &gen, uniform_real_distribution<double> &udist)
{
    double drift = (r - 0.5 * sigma * sigma);
    double sumPayoff = 0.0;
    double sumSqPayoff = 0.0;

    vector<vector<double>> Z(m);
    for (int i = 0; i < m; ++i)
    {
        Z[i].reserve(k);
    }

    vector<vector<double>> W(m + 1, vector<double>(k, 0.0));
    vector<vector<double>> S(m + 1, vector<double>(k, S0));

    for (int a = 0; a < n_eff; ++a)
    {
        for (int i = 0; i < m; ++i)
        {
            Z[i].clear();
            for (int j = 0; j < k; ++j)
            {
                Z[i].push_back(j);
            }
            shuffle(Z[i].begin(), Z[i].end(), gen);
        }

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                double U = udist(gen);
                double uniform_sample = (Z[i][j] + U) / k;
                Z[i][j] = inverseNormalCDF(uniform_sample);
            }
        }

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                W[i + 1][j] = (W[i][j] + sqrt(dt) * Z[i][j]);
            }
        }

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                S[i][j] = S0 * exp(drift * (i * dt) + sigma * W[i][j]);
            }
        }

        double batchTotalPayoff = 0.0;
        for (int j = 0; j < k; ++j)
        {
            double s_sum = 0;
            for (int i = 0; i <= m; ++i)
            {
                s_sum += S[i][j];
            }

            double payoff = callPayoff(s_sum / (m + 1), K);
            batchTotalPayoff += payoff;
        }

        double batchAvgPayoff = batchTotalPayoff / k;

        sumPayoff += batchAvgPayoff;
        sumSqPayoff += batchAvgPayoff * batchAvgPayoff;
    }

    double meanPrice = exp(-r * T) * (sumPayoff / n_eff);
    double varianceOfBatchMeans = (sumSqPayoff / n_eff) - (sumPayoff / n_eff) * (sumPayoff / n_eff);
    double estimatorVariance = varianceOfBatchMeans / n_eff;
    double stdError = sqrt(estimatorVariance) * exp(-r * T);

    cout << "--- LHS Monte Carlo ---" << endl;
    cout << "Estimated Price: " << meanPrice << endl;
    cout << "Std Error: " << stdError << endl;
}

vector<double> brownian_motion(int steps, double r, double sigma, double dt, mt19937 &gen, normal_distribution<double> &dist)
{
    vector<double> Z(steps);
    for (int i = 0; i < steps; ++i)
        Z[i] = dist(gen);

    vector<double> S(steps + 1, S0);
    double drift_term = (r - 0.5 * sigma * sigma) * dt;
    double vol_term = sigma * sqrt(dt);

    for (int i = 1; i <= steps; ++i)
    {
        S[i] = S[i - 1] * exp(drift_term + vol_term * Z[i - 1]);
    }
    return S;
}

void runStandardMonteCarlo(mt19937 &gen, normal_distribution<double> &dist)
{
    double sumPayoff = 0.0;
    double sumSqPayoff = 0.0;
    for (int a = 0; a < n_eff; ++a)
    {
        double batchTotalPayoff = 0.0;

        for (int j = 0; j < k; ++j)
        {
            vector<double> S = brownian_motion(m, r, sigma, dt, gen, dist);
            double S_avg = accumulate(S.begin(), S.end(), 0.0) / (m + 1);

            batchTotalPayoff += callPayoff(S_avg, K);
        }

        double batchAvgPayoff = batchTotalPayoff / k;

        sumPayoff += batchAvgPayoff;
        sumSqPayoff += batchAvgPayoff * batchAvgPayoff;
    }
    double meanPrice = exp(-r * T) * (sumPayoff / n_eff);
    double varianceOfBatchMeans = (sumSqPayoff / n_eff) - (sumPayoff / n_eff) * (sumPayoff / n_eff);
    double estimatorVariance = varianceOfBatchMeans / n_eff;
    double stdError = sqrt(estimatorVariance) * exp(-r * T);

    cout << "--- Standard Monte Carlo ---" << endl;
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
    cout << "Total Simulations: " << n << " (" << n_eff << " batches of " << k << ")" << endl
         << endl;

    runStandardMonteCarlo(gen, dist);
    cout << endl;
    runLHSMonteCarlo(gen, udist);

    return 0;
}