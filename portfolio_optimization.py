import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Portfolio Optimization Example
# -------------------------------

# Simulated daily returns for 3 assets
np.random.seed(42)
returns = pd.DataFrame({
    "Asset_A": np.random.normal(0.001, 0.02, 1000),
    "Asset_B": np.random.normal(0.0005, 0.015, 1000),
    "Asset_C": np.random.normal(0.002, 0.025, 1000)
})

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Number of portfolios to simulate
num_portfolios = 5000
results = np.zeros((3+3, num_portfolios))  # weights + return + volatility + sharpe

risk_free_rate = 0.0001

for i in range(num_portfolios):
    weights = np.random.random(3)
    weights /= np.sum(weights)
    
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    results[0:3, i] = weights
    results[3, i] = portfolio_return
    results[4, i] = portfolio_volatility
    results[5, i] = sharpe_ratio

# Convert results to DataFrame
results_df = pd.DataFrame(results.T, columns=['Asset_A','Asset_B','Asset_C','Return','Volatility','Sharpe'])

# Find portfolio with max Sharpe ratio
max_sharpe_port = results_df.iloc[results_df['Sharpe'].idxmax()]

print("Optimal Portfolio Weights:")
print(max_sharpe_port[['Asset_A','Asset_B','Asset_C']])
print("\nExpected Return:", max_sharpe_port['Return'])
print("Volatility:", max_sharpe_port['Volatility'])
print("Sharpe Ratio:", max_sharpe_port['Sharpe'])

# Plot Efficient Frontier
plt.figure(figsize=(10,6))
plt.scatter(results_df.Volatility, results_df.Return, c=results_df.Sharpe, cmap='viridis', s=10)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.show()
