#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
using namespace std;

double calculateMean(const vector<double> &data)
{
    return accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double calculateVariance(double mean, const vector<double> &data)
{
    double accum = 0.0;
    for (double d : data)
    {
        accum += (d - mean) * (d - mean);
    }
    return accum / data.size();
}

double calculateCovariance(double mean1, double mean2, const vector<double> &v1, const vector<double> &v2)
{
    double accum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i)
    {
        accum += (v1[i] - mean1) * (v2[i] - mean2);
    }
    return accum / v1.size();
}

void runStandardMonteCarlo(mt19937 &gen, int n)
{
    uniform_real_distribution<double> dist(0.0, 1.0);
    vector<double> Y;

    for (int i = 0; i < n; ++i)
    {
        double u = dist(gen);
        Y.push_back(exp(u * u));
    }

    double meanY = calculateMean(Y);
    double varianceY = calculateVariance(meanY, Y);
    double varianceEstimator = varianceY / n;
    double stdError = sqrt(varianceEstimator);

    cout << "--- Standard Monte Carlo ---" << endl;
    cout << "Estimated Mean: " << meanY << endl;
    cout << "Std Error: " << stdError << endl;
}

void runControlVariates(mt19937 &gen, int n)
{
    uniform_real_distribution<double> dist(0.0, 1.0);
    vector<double> Y, X;

    for (int i = 0; i < n; ++i)
    {
        double u = dist(gen);
        Y.push_back(exp(u * u));
        X.push_back(u * u);
    }

    double meanX = calculateMean(X);
    double meanY = calculateMean(Y);
    double muX = 1.0 / 3.0;

    double covXY = calculateCovariance(meanX, meanY, X, Y);
    double varX = calculateVariance(meanX, X);
    double b = covXY / varX;

    for (int i = 0; i < n; ++i)
    {
        Y[i] = Y[i] - b * (X[i] - muX);
    }

    double meanY_cv = meanY - b * (meanX - muX);
    double varianceY_cv = calculateVariance(meanY_cv, Y);
    double varianceEstimator = varianceY_cv / n;
    double stdError = sqrt(varianceEstimator);

    cout << "--- Control Variates ---" << endl;
    cout << "Estimated Mean: " << meanY_cv << endl;
    cout << "Std Error: " << stdError << endl;
}

int main()
{
    int n = 100000;
    random_device rd;
    mt19937 gen(rd());

    cout << fixed << setprecision(8);

    runStandardMonteCarlo(gen, n);
    cout << endl;
    runControlVariates(gen, n);

    return 0;
}