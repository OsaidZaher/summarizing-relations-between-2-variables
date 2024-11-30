import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(suppress=True, precision=2)

nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season and 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

print(nba_2010.head())
print(nba_2014.head())

# Extract points for Knicks and Nets in 2010
knicks_pts_10 = nba_2010.pts[nba_2010["fran_id"] == "Knicks"]
nets_pts_10 = nba_2010.pts[nba_2010["fran_id"] == "Nets"]


knicks_pts_14 = nba_2010.pts[nba_2014["fran_id"] == "Knicks"]
nets_pts_14 = nba_2010.pts[nba_2014["fran_id"] == "Nets"]

# Calculate absolute difference in mean points
diff_means_2010 = abs(knicks_pts_10.mean() - nets_pts_10.mean())
print(f"Absolute difference in mean points in 2010: {diff_means_2010}")

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(knicks_pts_10, alpha=0.8, density=True, label="Knicks")
plt.hist(nets_pts_10, alpha=0.8, density=True, label="Nets")

plt.legend()
plt.title("Points Distribution in 2010")
plt.xlabel("Points")
plt.ylabel("Density")

diff_means_2014 = abs(knicks_pts_14.mean() - nets_pts_14.mean())





# Create boxplot
plt.clf() 
sns.boxplot(data=nba_2010, x='fran_id', y='pts')
plt.title("Boxplot of Points by Team in 2010")
plt.xlabel("Team")
plt.ylabel("Points")
plt.show()

# Create contingency table for game results and locations
location_result_freq = pd.crosstab(nba_2010['game_result'], nba_2010['game_location'])
print(location_result_freq)

# Perform Chi-Square test
chi2, pval, dof, expected = chi2_contingency(location_result_freq)
print("Expected frequencies:\n", expected)
print("Chi-Square statistic:", chi2)

# Calculate covariance between forecast and point_diff
cov_matrix = np.cov(nba_2010['forecast'], nba_2010['point_diff'])
point_diff_forecast_cov = cov_matrix[0, 1]
print("Covariance between forecast and point_diff:", point_diff_forecast_cov)

# Calculate correlation between forecast and point_diff
point_diff_forecast_corr = pearsonr(nba_2010['forecast'], nba_2010['point_diff'])
print("Correlation between forecast and point_diff:", point_diff_forecast_corr)

# Create scatter plot of forecast vs. point_diff
plt.clf()  # Clear the previous plot
plt.scatter(nba_2010['forecast'], nba_2010['point_diff'])
plt.xlabel('Forecasted Win Probability')
plt.ylabel('Point Differential')
plt.title('Scatter Plot of Forecast vs. Point Differential')
plt.show()
