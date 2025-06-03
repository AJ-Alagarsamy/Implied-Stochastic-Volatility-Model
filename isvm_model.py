import numpy as np
import pandas as pd
from statsmodels.api import OLS
import matplotlib.pyplot as plt

def input_data_manually():
    """
    Input your option data directly in this function.
    Returns a DataFrame with columns: ['S', 'K', 'tau', 'iv', 'k']
    """
    # Example data - REPLACE THIS WITH YOUR ACTUAL DATA
    # Format: List of [S, K, tau (years), iv]
    your_data = [
        [84.75, 95, 0.53972602739, 0.59],
        [84.75, 95, 0.53972602739, 0.58],
        [84.75, 95, 0.53972602739, 0.57],
        [84.75, 95, 0.53972602739, 0.56],
        [84.75, 95, 0.53972602739, 0.55],
        [84.75, 95, 0.53972602739, 0.54],
        [84.75, 95, 0.53972602739, 0.53],
        [84.75, 95, 0.53972602739, 0.45],
        [84.75, 95, 0.53972602739, 0.44],
        [84.75, 95, 0.53972602739, 0.43],
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(your_data, columns=['S', 'K', 'tau', 'iv'])
    
    # Calculate log-moneyness (k = ln(K/S))
    df['k'] = np.log(df['K'] / df['S'])
    
    return df

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

def estimate_svm(iv_features, r=0.02, d=0.01, volatility_multiplier=1.0, leverage_multiplier=1.0):
    """Convert IV features to SV model coefficients with volatility adjustments"""
    S00, S01, S02, S10 = iv_features['S00'], iv_features['S01'], iv_features['S02'], iv_features['S10']
    
    # γ(v) = 2*Σ_0,0*Σ_0,1 with leverage adjustment
    gamma = 2 * S00 * S01 * leverage_multiplier
    
    # η(v) with volatility multiplier
    eta = np.sqrt(6 * S00**3 * S02 + 6 * gamma**2) * volatility_multiplier
    
    # μ(v) with slower mean reversion
    mu = (2*S10 - gamma*(d - r + gamma/4)/S00) * 0.7
    
    return {
        'mu': mu,
        'gamma': gamma,
        'eta': eta,
        'leverage_effect': gamma / np.sqrt(gamma**2 + eta**2),
        'volatility_regime': 'high' if volatility_multiplier > 1 else 'normal'
    }

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
    
    # Add volatility warning if in high regime
    if svm_params.get('volatility_regime', 'normal') == 'high':
        for ax in [ax1, ax2, ax3]:
            ax.annotate('HIGH VOLATILITY REGIME', 
                       xy=(0.5, 1.05), 
                       xycoords='axes fraction',
                       ha='center', 
                       color='red',
                       fontsize=12)
    
    plt.tight_layout()
    plt.show()

def main():
    # ======================
    # 1. Load Data
    # ======================
    print("Loading data...")
    df = input_data_manually()
    
    print("\nOption Data Preview:")
    print(df.head())
    
    # ======================
    # 2. Estimate IV Surface Characteristics
    # ======================
    print("\nEstimating IV features...")
    iv_features = estimate_iv_features(df)
    print("\nEstimated IV Features:")
    for k, v in iv_features.items():
        print(f"{k}: {v:.4f}")
    
    # ======================
    # 3. Nonparametric ISVM Estimation
    # ======================
    print("\nEstimating SV model parameters for volatile market...")
    svm_params = estimate_svm(
        iv_features,
        r=0.02,
        d=0.01,
        volatility_multiplier=2.0,  # Double the volatility of volatility
        leverage_multiplier=1.5     # Increase leverage effect by 50%
    )
    print("\nEstimated SV Model Parameters:")
    for k, v in svm_params.items():
        print(f"{k}: {v:.4f}")
    
    # ======================
    # 4. Visualization
    # ======================
    print("\nGenerating plots...")
    plot_results(df, iv_features, svm_params)

if __name__ == "__main__":
    main()