import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.api import OLS
from sklearn.kernel_ridge import KernelRidge  # Replacement for LocalLinearRegression
import matplotlib.pyplot as plt

# ======================
# 1. Simulate Fake Data
# ======================
def simulate_iv_surface(S0=100, r=0.02, d=0.01, days=30, n_strikes=10):
    """Generate synthetic implied volatility surface data"""
    np.random.seed(42)
    strikes = S0 * np.linspace(0.8, 1.2, n_strikes)
    maturities = np.linspace(15/365, 60/365, 5)
    
    data = []
    for tau in maturities:
        for K in strikes:
            k = np.log(K/S0)
            # Synthetic IV surface with skew and term structure
            iv = 0.2 + 0.1*k - 0.3*k**2 + 0.05*tau  
            data.append([S0, K, tau, k, iv])
    
    return pd.DataFrame(data, columns=['S', 'K', 'tau', 'k', 'iv'])

df = simulate_iv_surface()
print("Simulated Option Data:")
print(df.head())

# ======================
# 2. Estimate IV Surface Characteristics
# ======================
def estimate_iv_features(df):
    """Fit polynomial regression to get Σ_0,0, Σ_0,1, Σ_0,2, Σ_1,0"""
    X = pd.DataFrame({
        'const': 1,
        'tau': df['tau'],
        'tau2': df['tau']**2,
        'k': df['k'],
        'tau_k': df['tau']*df['k'],
        'k2': df['k']**2
    })
    model = OLS(df['iv'], X).fit()
    
    return {
        'S00': model.params['const'],       # ATM level (Σ_0,0)
        'S01': model.params['k'],           # Moneyness slope (Σ_0,1)
        'S02': 2*model.params['k2'],        # Moneyness convexity (Σ_0,2)
        'S10': model.params['tau'],         # Term structure slope (Σ_1,0)
    }

iv_features = estimate_iv_features(df)
print("\nEstimated IV Features:")
for k, v in iv_features.items():
    print(f"{k}: {v:.4f}")

# ======================
# 3. Nonparametric ISVM Estimation
# ======================
def estimate_svm(iv_features, r=0.02, d=0.01):
    """Convert IV features to SV model coefficients"""
    S00, S01, S02, S10 = iv_features['S00'], iv_features['S01'], iv_features['S02'], iv_features['S10']
    
    # γ(v) = 2*Σ_0,0*Σ_0,1
    gamma = 2 * S00 * S01
    
    # η(v) = sqrt[6Σ_0,0³Σ_0,2 + 6γ²] (simplified)
    eta = np.sqrt(6 * S00**3 * S02 + 6 * gamma**2)
    
    # μ(v) simplified calculation
    mu = 2*S10 - gamma*(d - r + gamma/4)/S00
    
    return {
        'mu': mu,
        'gamma': gamma,
        'eta': eta,
        'leverage_effect': gamma / np.sqrt(gamma**2 + eta**2)
    }

svm_params = estimate_svm(iv_features)
print("\nEstimated SV Model Parameters:")
for k, v in svm_params.items():
    print(f"{k}: {v:.4f}")

# ======================
# 4. Visualization
# ======================
def plot_results(df, iv_features, svm_params):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # IV Surface
    sc = ax1.scatter(df['k'], df['iv'], c=df['tau'], cmap='viridis')
    ax1.set_title("Implied Volatility Surface")
    ax1.set_xlabel("Log-Moneyness (k)")
    ax1.set_ylabel("IV")
    plt.colorbar(sc, ax=ax1, label='Time to Maturity (τ)')
    
    # IV Features
    features = list(iv_features.keys())
    values = list(iv_features.values())
    ax2.bar(features, values)
    ax2.set_title("IV Surface Characteristics")
    ax2.set_ylabel("Value")
    
    # SV Parameters
    params = list(svm_params.keys())
    param_values = list(svm_params.values())
    ax3.bar(params, param_values)
    ax3.set_title("Stochastic Volatility Parameters")
    
    plt.tight_layout()
    plt.savefig('isvm_results.png')
    plt.show()

plot_results(df, iv_features, svm_params)