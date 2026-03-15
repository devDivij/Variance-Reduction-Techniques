# Monte Carlo Variance Reduction Techniques

C++ implementations of variance reduction methods for pricing financial options using Monte Carlo simulation.

## Methods Implemented

### 1. **Antithetic Variates**
Uses paired samples (Z, -Z) to exploit negative correlation and reduce variance.
- **Variance Reduction Factor observed:** 1.25x

### 2. **Control Variates**
Adjusts estimates using a correlated variable with known mean.
- **Variance Reduction Factor observed:** 7.72x

### 3. **Stratified Sampling**
Partitions probability space into strata and samples from each.
- **Variance Reduction Factor observed:** 11.1x

### 4. **Latin Hypercube Sampling (LHS)**
Multi-dimensional stratification with better space-filling properties.
- **Variance Reduction Factor observed:** 1.6x

### 5. **Importance Sampling**
Samples from shifted distribution centered near important regions.
- **Variance Reduction Factor observed:** 7.3x

## Quick Start

**Compile:**
```bash
g++ -std=c++11 -O3 -o method method_name.cpp
```

**Run:**
```bash
./method
```

## Key Formulas

**European Call Option:**
$$C(0) = e^{-rT} \mathbb{E}[\max(S(T) - K, 0)]$$

$$S(T) = S_0 \exp\left(\left(r - \frac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}Z\right)$$

**Variance Reduction Factor:**
```
VRF = Var(Standard MC) / Var(Variance Reduction Method)
```

## References

- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*