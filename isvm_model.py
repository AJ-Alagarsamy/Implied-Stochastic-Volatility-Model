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
        [0.7375, 0.95, 0.7918, 0.4617307305],
        [0.76, 0.95, 0.7808, 0.4259649054],
        [0.745, 0.95, 0.7781, 0.4459926271],
        [0.7075, 0.95, 0.7753, 0.7095802862],
        [0.72, 0.95, 0.7726, 0.5230578182],
        [0.7, 0.95, 0.7699, 0.5554256522],
        [0.6975, 0.95, 0.7616, 0.5863223041],
        [0.71, 0.95, 0.7589, 0.5666695983],
        [0.7125, 0.95, 0.7562, 0.5572853822],
        [0.73, 0.95, 0.7534, 0.5284698409],
        [0.735, 0.95, 0.7507, 0.5213566326],
        [0.74, 0.95, 0.7425, 0.6093807174],
        [0.765, 0.95, 0.7397, 0.5704813832],
        [0.755, 0.95, 0.737, 0.5904565477],
        [0.755, 0.95, 0.7342, 0.7709006102],
        [0.7625, 0.95, 0.7315, 0.8172667147],
        [0.805, 0.95, 0.7233, 0.7884570188],
        [0.84, 0.95, 0.7205, 0.7081118713],
        [0.8425, 0.95, 0.7178, 0.7046025648],
        [0.885, 0.95, 0.7151, 0.58000218],
        [0.955, 0.95, 0.7123, 0.4707757872],
        [0.9525, 0.95, 0.7041, 0.509628914],
        [0.94, 0.95, 0.7014, 0.4822188385],
        [0.945, 0.95, 0.6986, 0.5380890207],
        [0.9425, 0.95, 0.6959, 0.4984481875],
        [0.96, 0.95, 0.6932, 0.452417545],
        [0.965, 0.95, 0.6849, 0.4455955298],
        [0.965, 0.95, 0.6822, 0.4761675777],
        [0.95, 0.95, 0.6795, 0.4937282598],
        [0.9675, 0.95, 0.6767, 0.5083943892],
        [0.98, 0.95, 0.674, 0.6228693305],
        [0.9825, 0.95, 0.6658, 0.6097748839],
        [0.9775, 0.95, 0.663, 0.6795012194],
        [0.97875, 0.95, 0.6712, 0.4818],
        [0.985, 0.95, 0.6685, 0.3941],
        [1.02, 0.95, 0.6575, 0.3414],
        [1.0625, 0.95, 0.6548, 0.5023],
        [1.06, 0.95, 0.6521, 0.4888],
        [1.06, 0.95, 0.6493, 0.5391],
        [1.0675, 0.95, 0.6466, 0.4426],
        [1.1025, 0.95, 0.6384, 0.4534],
        [1.105, 0.95, 0.6356, 0.5176],
        [1.1175, 0.95, 0.6329, 0.6227],
        [1.08, 0.95, 0.6301, 0.5829],
        [1.085, 0.95, 0.6274, 0.5855],
        [1.0725, 0.95, 0.6192, 0.5187],
        [1.065, 0.95, 0.6164, 0.4488],
        [1.075, 0.95, 0.6137, 0.4201],
        [1.0875, 0.95, 0.6109, 0.4052],
        [1.105, 0.95, 0.6082, 0.7746],
        [1.0075, 0.95, 0.6, 0.8632],
        [1.0225, 0.95, 0.5973, 0.4269],
        [1.0275, 0.95, 0.5945, 0.4756],
        [1.04, 0.95, 0.5918, 0.4605],
        [1.06, 0.95, 0.589, 0.5745],
        [1.03, 0.95, 0.5808, 0.6964],
        [1.04, 0.95, 0.5781, 0.5127],
        [1.04, 0.95, 0.5753, 0.6478],
        [1.0025, 0.95, 0.5726, 0.8141],
        [0.9275, 0.95, 0.5699, 0.7688],
        [0.89, 0.95, 0.5589, 0.855],
        [0.9425, 0.95, 0.5562, 0.5425],
        [0.8895, 0.95, 0.5425, 0.8141],
        [0.8475, 0.95, 0.5425, 0.7688],
        [0.8675, 0.95, 0.5397, 0.855],
        [0.9, 0.95, 0.5369, 0.5425],
        [0.9001, 0.95, 0.5342465753, 0.5667]
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
    params = [p for p in svm_params.keys() if p != 'volatility_regime']
    param_values = [svm_params[p] for p in params]
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
        if isinstance(v, str):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")
    
    # ======================
    # 4. Visualization
    # ======================
    print("\nGenerating plots...")
    plot_results(df, iv_features, svm_params)

if __name__ == "__main__":
    main()