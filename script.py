import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings



# --- (Previous code for data loading, cleaning, aggregation remains the same) ---
# Ignore warnings
warnings.filterwarnings('ignore')

# Load the dataframe
try:
    # Load the dataframe from the uploaded file
    df = pd.read_csv('combined.csv')
    print("Successfully loaded combined.csv")
except FileNotFoundError:
    print("Error: 'combined.csv' not found. Please ensure the file is uploaded.")
    exit() # Exit if the file isn't found
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# --- Data Cleaning ---
if 'Unnamed: 4' in df.columns:
    df = df.drop(columns=['Unnamed: 4'])

urban_col_name = None
for col in df.columns:
     # Be more specific, check for common variations
    if 'Urban' in col and ('MPCE' in col or 'average' in col):
        urban_col_name = col
        break

if urban_col_name:
    df = df.rename(columns={
        # Try to find the rural column more dynamically too
        next((c for c in df.columns if 'Rural' in c and ('MPCE' in c or 'average' in c)), 'average MPCE (Rs.) Rural'): 'MPCE_Rural',
        urban_col_name: 'MPCE_Urban',
        next((c for c in df.columns if 'Year' in c or 'Financial' in c), 'Year'): 'Financial_Year'
    })
    print("Renamed columns using dynamic detection.")
else:
    print("Warning: Could not definitively identify the Urban MPCE column. Using default fallback names.")
    # Use more robust potential default names if dynamic detection fails
    df = df.rename(columns={
        'average MPCE (Rs.) Rural': 'MPCE_Rural',
        'average MPCE (Rs.) Urban': 'MPCE_Urban', # Default guess
        'Year': 'Financial_Year'
    })

# Handle potential missing columns after renaming attempts
required_cols = ['Financial_Year', 'MPCE_Rural', 'MPCE_Urban']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing required columns after renaming: {missing_cols}. Please check the CSV headers.")
    print("Current columns:", df.columns.tolist())
    exit()


df['Financial_Year'] = df['Financial_Year'].astype(str)
# Robustly extract the starting year, handle formats like '2007-08', '2007-2008', '2007'
df['Year'] = df['Financial_Year'].str.extract(r'^(\d{4})').iloc[:, 0]
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=['Year'])
df['Year'] = df['Year'].astype(int)

df['MPCE_Rural'] = pd.to_numeric(df['MPCE_Rural'], errors='coerce')
df['MPCE_Urban'] = pd.to_numeric(df['MPCE_Urban'], errors='coerce')

# Drop rows where *any* of the essential columns are missing
df = df.dropna(subset=['MPCE_Rural', 'MPCE_Urban', 'Year'])
print(f"Cleaned data has {len(df)} rows.")

# --- Aggregate Data Yearly ---
yearly_avg_mpce = df.groupby('Year')[['MPCE_Rural', 'MPCE_Urban']].mean().reset_index()
yearly_avg_mpce = yearly_avg_mpce.sort_values('Year')

print("\nHistorical Average MPCE across States:")
if not yearly_avg_mpce.empty:
    print(yearly_avg_mpce.to_markdown(index=False, numalign="left", stralign="left", floatfmt=".2f"))
else:
    print("No historical data available after cleaning and aggregation.")
    exit()

# --- Define Target Years and Prepare Data ---
prediction_years_num = list(range(2012, 2022)) # Years 2012 to 2021 inclusive
all_predictions_list = [] # List to store prediction dictionaries for each year

print(f"\nPredicting/Interpolating average MPCE for the years: {prediction_years_num}")

# --- Loop Through Each Target Year for Prediction ---
for target_year_num in prediction_years_num:
    target_year_str = f"{target_year_num}-{str(target_year_num + 1)[-2:]}"
    target_year_input = np.array([[target_year_num]])

    print(f"\n--- Processing Year: {target_year_str} ({target_year_num}) ---")

    # Check if this year exists in the historical data
    actual_target_year_data = yearly_avg_mpce[yearly_avg_mpce['Year'] == target_year_num]
    has_actual_data = not actual_target_year_data.empty
    if has_actual_data:
        print(f"Note: Actual data exists for {target_year_num}. Models will be trained excluding this year.")
    else:
         print(f"Note: No actual data for {target_year_num} in historical set. Predicting based on other years.")

    # Prepare training data by excluding the current target year
    training_data = yearly_avg_mpce[yearly_avg_mpce['Year'] != target_year_num].copy()

    if training_data.empty:
        print(f"\nError: No historical data available after excluding {target_year_num}. Cannot train models for this year.")
        # Add NaNs for this year and continue
        nan_preds = {'Year_Num': target_year_num, 'Year': target_year_str}
        pred_cols = ['Interp_Linear_Rural', 'Interp_Linear_Urban', 'Interp_Spline_Rural', 'Interp_Spline_Urban',
                     'Linear_Avg_Rural', 'Linear_Avg_Urban', 'Poly_Avg_Rural', 'Poly_Avg_Urban',
                     'SVR_Avg_Rural', 'SVR_Avg_Urban', 'GPR_Avg_Rural', 'GPR_Avg_Urban']
        for col in pred_cols:
            nan_preds[col] = np.nan
        all_predictions_list.append(nan_preds)
        continue # Skip to the next year

    X_train = training_data[['Year']]
    y_train_rural = training_data['MPCE_Rural']
    y_train_urban = training_data['MPCE_Urban']

    print(f"Using data from {len(training_data)} year(s) for model training/interpolation.")

    # --- Initialize Predictions Dictionary for the current target year ---
    current_predictions = {'Year_Num': target_year_num, 'Year': target_year_str}

    # --- Method 1 & 2: Interpolation (Linear and Spline) ---
    print("\nAttempting Interpolation...")
    years_before = training_data[training_data['Year'] < target_year_num]['Year'].max()
    years_after = training_data[training_data['Year'] > target_year_num]['Year'].min()
    interp_possible = False
    if pd.notna(years_before) and pd.notna(years_after):
        interp_possible = True
        print(f"Found surrounding years in training data: {int(years_before)} and {int(years_after)}")

        # Get data for interpolation including the target year (value will be NaN initially)
        interp_df_slice = training_data[(training_data['Year'] == years_before) | (training_data['Year'] == years_after)].copy()
        target_row = pd.DataFrame([{'Year': target_year_num, 'MPCE_Rural': np.nan, 'MPCE_Urban': np.nan}]) # Explicitly add NaN row
        interp_df_full = pd.concat([interp_df_slice, target_row], ignore_index=True).sort_values('Year')
        interp_df_full = interp_df_full.set_index('Year')


        # Linear Interpolation
        interp_linear = interp_df_full.interpolate(method='index', limit_direction='both')
        pred_interp_linear_rural = interp_linear.loc[target_year_num, 'MPCE_Rural']
        pred_interp_linear_urban = interp_linear.loc[target_year_num, 'MPCE_Urban']
        current_predictions['Interp_Linear_Rural'] = np.maximum(0, pred_interp_linear_rural)
        current_predictions['Interp_Linear_Urban'] = np.maximum(0, pred_interp_linear_urban)
        print(f"Linear Interpolation Prediction (Rural): {current_predictions['Interp_Linear_Rural']:.2f}")
        print(f"Linear Interpolation Prediction (Urban): {current_predictions['Interp_Linear_Urban']:.2f}")

        # Spline Interpolation (using only surrounding points for stability if few points overall)
        try:
             # Use only the immediately surrounding points for spline interpolation
            spline_data_strict = training_data[(training_data['Year'] == years_before) | (training_data['Year'] == years_after)]
            if len(spline_data_strict) >= 2: # Need at least two points
                spline_order = 1 # Linear spline with just two points
                # If more points are available *between* years_before and years_after in the training data, consider using them for a higher order spline
                # relevant_spline_data = training_data[(training_data['Year'] >= years_before) & (training_data['Year'] <= years_after)]
                # if len(relevant_spline_data) >= 4:
                #     spline_order = 3
                # elif len(relevant_spline_data) == 3:
                #     spline_order = 2

                # Using interp1d on the two surrounding points:
                f_spline_rural = interp1d(spline_data_strict['Year'], spline_data_strict['MPCE_Rural'], kind=spline_order, fill_value="extrapolate", bounds_error=False)
                f_spline_urban = interp1d(spline_data_strict['Year'], spline_data_strict['MPCE_Urban'], kind=spline_order, fill_value="extrapolate", bounds_error=False)

                pred_interp_spline_rural = f_spline_rural(target_year_num)
                pred_interp_spline_urban = f_spline_urban(target_year_num)
                current_predictions['Interp_Spline_Rural'] = np.maximum(0, pred_interp_spline_rural.item()) # .item() to get scalar
                current_predictions['Interp_Spline_Urban'] = np.maximum(0, pred_interp_spline_urban.item()) # .item() to get scalar
                print(f"Spline (Order {spline_order}) Interpolation Prediction (Rural): {current_predictions['Interp_Spline_Rural']:.2f}")
                print(f"Spline (Order {spline_order}) Interpolation Prediction (Urban): {current_predictions['Interp_Spline_Urban']:.2f}")
            else:
                 raise ValueError("Not enough surrounding data points for spline interpolation.")

        except Exception as e:
            print(f"Could not perform Spline interpolation for {target_year_num}: {e}")
            current_predictions['Interp_Spline_Rural'] = np.nan
            current_predictions['Interp_Spline_Urban'] = np.nan
    else:
        print(f"Could not find data points both before and after {target_year_num}. Skipping interpolation.")
        current_predictions['Interp_Linear_Rural'], current_predictions['Interp_Linear_Urban'] = np.nan, np.nan
        current_predictions['Interp_Spline_Rural'], current_predictions['Interp_Spline_Urban'] = np.nan, np.nan

    # --- Regression Models ---
    print("\nFitting Regression Models...")
    if len(training_data) >= 2:
        # Method 3: Linear Regression
        try:
            model_linear_rural = LinearRegression().fit(X_train, y_train_rural)
            model_linear_urban = LinearRegression().fit(X_train, y_train_urban)
            pred_linear_rural = model_linear_rural.predict(target_year_input)
            pred_linear_urban = model_linear_urban.predict(target_year_input)
            current_predictions['Linear_Avg_Rural'] = np.maximum(0, pred_linear_rural)[0]
            current_predictions['Linear_Avg_Urban'] = np.maximum(0, pred_linear_urban)[0]
            print(f"Linear Regression Prediction (Rural): {current_predictions['Linear_Avg_Rural']:.2f}")
            print(f"Linear Regression Prediction (Urban): {current_predictions['Linear_Avg_Urban']:.2f}")
        except Exception as e:
            print(f"Could not perform Linear Regression for {target_year_num}: {e}")
            current_predictions['Linear_Avg_Rural'], current_predictions['Linear_Avg_Urban'] = np.nan, np.nan


        # Method 4: Polynomial Regression (Degree 2)
        if len(training_data) >= 3:
             try:
                poly_pipeline_rural = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                poly_pipeline_urban = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                poly_pipeline_rural.fit(X_train, y_train_rural)
                poly_pipeline_urban.fit(X_train, y_train_urban)
                pred_poly_rural = poly_pipeline_rural.predict(target_year_input)
                pred_poly_urban = poly_pipeline_urban.predict(target_year_input)
                current_predictions['Poly_Avg_Rural'] = np.maximum(0, pred_poly_rural)[0]
                current_predictions['Poly_Avg_Urban'] = np.maximum(0, pred_poly_urban)[0]
                print(f"Polynomial (Deg 2) Prediction (Rural): {current_predictions['Poly_Avg_Rural']:.2f}")
                print(f"Polynomial (Deg 2) Prediction (Urban): {current_predictions['Poly_Avg_Urban']:.2f}")
             except Exception as e:
                print(f"Could not perform Polynomial Regression for {target_year_num}: {e}")
                current_predictions['Poly_Avg_Rural'], current_predictions['Poly_Avg_Urban'] = np.nan, np.nan
        else:
            print("Insufficient data points (need >= 3 excluding target year) for Polynomial Regression.")
            current_predictions['Poly_Avg_Rural'], current_predictions['Poly_Avg_Urban'] = np.nan, np.nan

        # Method 5: Support Vector Regression (SVR)
        try:
            svr_pipeline_rural = make_pipeline(StandardScaler(), SVR()) # Default kernel is RBF
            svr_pipeline_urban = make_pipeline(StandardScaler(), SVR())
            svr_pipeline_rural.fit(X_train, y_train_rural)
            svr_pipeline_urban.fit(X_train, y_train_urban)
            pred_svr_rural = svr_pipeline_rural.predict(target_year_input)
            pred_svr_urban = svr_pipeline_urban.predict(target_year_input)
            current_predictions['SVR_Avg_Rural'] = np.maximum(0, pred_svr_rural)[0]
            current_predictions['SVR_Avg_Urban'] = np.maximum(0, pred_svr_urban)[0]
            print(f"SVR Prediction (Rural): {current_predictions['SVR_Avg_Rural']:.2f}")
            print(f"SVR Prediction (Urban): {current_predictions['SVR_Avg_Urban']:.2f}")
        except Exception as e:
            print(f"Could not perform SVR prediction for {target_year_num}: {e}")
            current_predictions['SVR_Avg_Rural'], current_predictions['SVR_Avg_Urban'] = np.nan, np.nan

        # Method 6: Gaussian Process Regression (GPR)
        # Define kernel (adjust parameters as needed, maybe based on data scale)
        # Increased noise level bounds slightly and default length scale
        kernel = C(1.0, (1e-3, 1e4)) * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 1e3)) \
              + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
        try:
            gpr_rural = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
            gpr_urban = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
            gpr_rural.fit(X_train, y_train_rural)
            gpr_urban.fit(X_train, y_train_urban)
            pred_gpr_rural, std_gpr_rural = gpr_rural.predict(target_year_input, return_std=True)
            pred_gpr_urban, std_gpr_urban = gpr_urban.predict(target_year_input, return_std=True)
            current_predictions['GPR_Avg_Rural'] = np.maximum(0, pred_gpr_rural)[0]
            current_predictions['GPR_Avg_Urban'] = np.maximum(0, pred_gpr_urban)[0]
            # Optional: Store std deviation if needed later, but not printing by default here
            # current_predictions['GPR_Std_Rural'] = std_gpr_rural[0]
            # current_predictions['GPR_Std_Urban'] = std_gpr_urban[0]
            print(f"GPR Prediction (Rural): {current_predictions['GPR_Avg_Rural']:.2f} (Std: {std_gpr_rural[0]:.2f})")
            print(f"GPR Prediction (Urban): {current_predictions['GPR_Avg_Urban']:.2f} (Std: {std_gpr_urban[0]:.2f})")
        except Exception as e:
            print(f"Could not perform GPR prediction for {target_year_num}: {e}")
            current_predictions['GPR_Avg_Rural'], current_predictions['GPR_Avg_Urban'] = np.nan, np.nan

    else:
        print("Insufficient historical data points (need >= 2 excluding target year) for regression models.")
        current_predictions['Linear_Avg_Rural'], current_predictions['Linear_Avg_Urban'] = np.nan, np.nan
        current_predictions['Poly_Avg_Rural'], current_predictions['Poly_Avg_Urban'] = np.nan, np.nan
        current_predictions['SVR_Avg_Rural'], current_predictions['SVR_Avg_Urban'] = np.nan, np.nan
        current_predictions['GPR_Avg_Rural'], current_predictions['GPR_Avg_Urban'] = np.nan, np.nan

    # --- Append current year's predictions to the list ---
    all_predictions_list.append(current_predictions)


# --- Consolidate All Predictions ---
predictions_all_df = pd.DataFrame(all_predictions_list)
# Convert Year_Num to integer if it's not already, before sorting
predictions_all_df['Year_Num'] = predictions_all_df['Year_Num'].astype(int)
predictions_all_df = predictions_all_df.sort_values('Year_Num') # Ensure sorted by year

# Define desired column order for the final table
display_cols = [
    'Year', 'Interp_Linear_Rural', 'Interp_Linear_Urban', 'Interp_Spline_Rural', 'Interp_Spline_Urban',
    'Linear_Avg_Rural', 'Linear_Avg_Urban', 'Poly_Avg_Rural', 'Poly_Avg_Urban',
    'SVR_Avg_Rural', 'SVR_Avg_Urban', 'GPR_Avg_Rural', 'GPR_Avg_Urban'
]
# Filter out columns that might not have been generated if models failed
existing_display_cols = [col for col in display_cols if col in predictions_all_df.columns]
predictions_display_df = predictions_all_df[existing_display_cols].copy()

print(f"\n--- Summary of Predicted/Interpolated Average MPCE (Rs.) for Years {prediction_years_num[0]}-{prediction_years_num[-1]} ---")
if not predictions_display_df.empty:
    print(predictions_display_df.round(2).to_markdown(index=False, numalign="left", stralign="left", floatfmt=".2f"))
else:
    print("No predictions were generated.")

# --- Comparison with Actual Data (Merge actual data where available) ---
# Merge historical data with predictions based on the numeric year for robustness
# Ensure yearly_avg_mpce['Year'] is integer for merging
yearly_avg_mpce['Year'] = yearly_avg_mpce['Year'].astype(int)

comparison_df = pd.merge(
    predictions_all_df, # Use the df with Year_Num (already int)
    yearly_avg_mpce.rename(columns={'MPCE_Rural': 'Actual_Rural', 'MPCE_Urban': 'Actual_Urban'})[['Year', 'Actual_Rural', 'Actual_Urban']],
    left_on='Year_Num', # Merge based on the numeric year
    right_on='Year',
    how='left'
)
# Clean up merge columns if 'Year' column from yearly_avg_mpce was added
if 'Year_y' in comparison_df.columns:
    comparison_df = comparison_df.drop(columns=['Year_y'])
if 'Year_x' in comparison_df.columns:
     comparison_df = comparison_df.rename(columns={'Year_x': 'Year'}) # Keep the original Year string 'YYYY-YY'


# Reorder columns for comparison table
# Start with 'Year' (string format), then prediction columns, then actuals
comp_cols_order = ['Year'] + \
                  [col for col in existing_display_cols if col != 'Year'] + \
                  ['Actual_Rural', 'Actual_Urban'] # Add Actuals at the end

# Filter comparison_df to only include columns that actually exist
final_comp_cols = [col for col in comp_cols_order if col in comparison_df.columns]
comparison_df = comparison_df[final_comp_cols]


print("\n--- Comparison Table (Predicted/Interpolated vs Actual where available) ---")
if not comparison_df.empty:
    print(comparison_df.round(2).to_markdown(index=False, numalign="left", stralign="left", floatfmt=".2f", missingval="N/A"))
else:
    print("Could not generate comparison table.")

# --- Visualization: Individual Plots for Each Method ---
print("\n--- Generating Individual Prediction Method Visualizations ---")

if not yearly_avg_mpce.empty and not predictions_all_df.empty:
    # Ensure 'Year' in yearly_avg_mpce is numeric for plotting
    yearly_avg_mpce['Year'] = pd.to_numeric(yearly_avg_mpce['Year'], errors='coerce').dropna()
    yearly_avg_mpce = yearly_avg_mpce.dropna(subset=['Year']) # Drop rows where conversion failed

    # Ensure Year_Num is numeric in predictions_all_df
    predictions_all_df['Year_Num'] = pd.to_numeric(predictions_all_df['Year_Num'], errors='coerce')
    predictions_all_df = predictions_all_df.dropna(subset=['Year_Num']) # Drop rows where conversion failed


    # Define methods and their corresponding column prefixes and plot labels
    prediction_methods = {
        'Interp_Linear': 'Interp Linear',
        'Interp_Spline': 'Interp Spline',
        'Linear_Avg': 'Linear Reg',
        'Poly_Avg': 'Poly Reg',
        'SVR_Avg': 'SVR',
        'GPR_Avg': 'GPR'
    }

    # Define consistent colors for historical data
    hist_color_rural = 'blue'
    hist_color_urban = 'green'

    # --- Calculate Common X-axis Ticks ---
    # Ensure all years used for ticks are numeric integers
    hist_years = pd.to_numeric(yearly_avg_mpce['Year'], errors='coerce').dropna().astype(int).unique()
    pred_years = pd.to_numeric(predictions_all_df['Year_Num'], errors='coerce').dropna().astype(int).unique()
    all_plot_years_num = sorted(list(set(hist_years) | set(pred_years)))
    selected_ticks = []
    if all_plot_years_num:
        if len(all_plot_years_num) > 15:
            step_comb = max(1, len(all_plot_years_num)//12) # Ensure step is at least 1
            selected_ticks = all_plot_years_num[::step_comb]
        else:
            selected_ticks = all_plot_years_num

    # Loop through each prediction method to create individual plots
    for method_prefix, method_label in prediction_methods.items():
        rural_col = f'{method_prefix}_Rural'
        urban_col = f'{method_prefix}_Urban'

        # Check if prediction columns exist and have data
        has_rural_preds = rural_col in predictions_all_df.columns and predictions_all_df[rural_col].notna().any()
        has_urban_preds = urban_col in predictions_all_df.columns and predictions_all_df[urban_col].notna().any()

        if not has_rural_preds and not has_urban_preds:
            print(f"Skipping plot for {method_label}: No prediction data available.")
            continue

        print(f"Generating plot for: {method_label}")
        fig_ind, axes_ind = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig_ind.suptitle(f'Historical MPCE and {method_label} Predictions', fontsize=14, y=1.0) # Adjusted y for title

        # --- Rural Individual Plot ---
        ax_r = axes_ind[0]
        ax_r.plot(yearly_avg_mpce['Year'], yearly_avg_mpce['MPCE_Rural'], marker='o', linestyle='-', color=hist_color_rural, label='Historical Avg Rural', alpha=0.7, markersize=5)
        if has_rural_preds:
            plot_data_rural = predictions_all_df.dropna(subset=[rural_col])
            ax_r.plot(plot_data_rural['Year_Num'], plot_data_rural[rural_col], marker='X', markersize=8, color='red', linestyle='None', label=f'{method_label} Pred.')
        ax_r.set_title(f'Average Rural MPCE vs. {method_label}')
        ax_r.set_ylabel('Average MPCE (Rs.)')
        ax_r.legend(fontsize='small', loc='upper left')
        ax_r.grid(True, linestyle=':', alpha=0.6)

        # --- Urban Individual Plot ---
        ax_u = axes_ind[1]
        ax_u.plot(yearly_avg_mpce['Year'], yearly_avg_mpce['MPCE_Urban'], marker='o', linestyle='-', color=hist_color_urban, label='Historical Avg Urban', alpha=0.7, markersize=5)
        if has_urban_preds:
            plot_data_urban = predictions_all_df.dropna(subset=[urban_col])
            ax_u.plot(plot_data_urban['Year_Num'], plot_data_urban[urban_col], marker='X', markersize=8, color='purple', linestyle='None', label=f'{method_label} Pred.')
        ax_u.set_title(f'Average Urban MPCE vs. {method_label}')
        ax_u.set_xlabel('Year')
        ax_u.set_ylabel('Average MPCE (Rs.)')
        ax_u.legend(fontsize='small', loc='upper left')
        ax_u.grid(True, linestyle=':', alpha=0.6)

        # Set x-axis ticks for the individual plot
        if selected_ticks:
            ax_u.set_xticks(selected_ticks)
            plt.setp(ax_u.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout
        plt.show() # Show the individual plot

else:
    print("\nSkipping individual visualizations: No historical data or no predictions generated.")


# --- Visualization: Combined Plot showing Historical and Predicted Data ---
print("\n--- Generating Combined Visualization for Predicted Years ---")
if not yearly_avg_mpce.empty and not predictions_all_df.empty:
    fig_comb, axes_comb = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig_comb.suptitle(f'Historical MPCE and All Predictions for {int(pred_years.min())}-{int(pred_years.max())}', fontsize=16, y=1.0) # Use actual predicted range

    # --- Rural Combined Plot ---
    # Plot Historical Data
    axes_comb[0].plot(yearly_avg_mpce['Year'], yearly_avg_mpce['MPCE_Rural'], marker='o', linestyle='-', color=hist_color_rural, label='Historical Avg Rural', alpha=0.7, markersize=6)

    # Plot Predicted Data for each method
    plot_markers = {
        'Interp_Linear': ('^', 'cyan', 'Interp Linear'),
        'Interp_Spline': ('v', 'lime', 'Interp Spline'),
        'Linear_Avg': ('x', 'orange', 'Linear Reg'),
        'Poly_Avg': ('+', 'magenta', 'Poly Reg'),
        'SVR_Avg': ('s', 'brown', 'SVR'),
        'GPR_Avg': ('d', 'gold', 'GPR')
    }
    for method, (marker, color, label) in plot_markers.items():
        col_name = f'{method}_Rural'
        # Check if the column exists and has non-NA values to plot
        if col_name in predictions_all_df.columns and predictions_all_df[col_name].notna().any():
            plot_data = predictions_all_df.dropna(subset=[col_name])
            axes_comb[0].plot(plot_data['Year_Num'], plot_data[col_name], marker=marker, markersize=8, color=color, linestyle='None', label=f'{label} Pred.')

    axes_comb[0].set_title('Average Rural MPCE')
    axes_comb[0].set_ylabel('Average MPCE (Rs.)')
    axes_comb[0].legend(fontsize='small', loc='upper left', ncol=2) # Adjust legend position/columns
    axes_comb[0].grid(True, linestyle=':', alpha=0.6)

    # --- Urban Combined Plot ---
     # Plot Historical Data
    axes_comb[1].plot(yearly_avg_mpce['Year'], yearly_avg_mpce['MPCE_Urban'], marker='o', linestyle='-', color=hist_color_urban, label='Historical Avg Urban', alpha=0.7, markersize=6)

    # Plot Predicted Data for each method
    for method, (marker, color, label) in plot_markers.items():
        col_name = f'{method}_Urban'
        if col_name in predictions_all_df.columns and predictions_all_df[col_name].notna().any():
            plot_data = predictions_all_df.dropna(subset=[col_name])
            axes_comb[1].plot(plot_data['Year_Num'], plot_data[col_name], marker=marker, markersize=8, color=color, linestyle='None', label=f'{label} Pred.')

    axes_comb[1].set_title('Average Urban MPCE')
    axes_comb[1].set_xlabel('Year')
    axes_comb[1].set_ylabel('Average MPCE (Rs.)')
    axes_comb[1].legend(fontsize='small', loc='upper left', ncol=2) # Adjust legend position/columns
    axes_comb[1].grid(True, linestyle=':', alpha=0.6)

    # Set x-axis ticks for combined plot using pre-calculated ticks
    if selected_ticks:
        axes_comb[1].set_xticks(selected_ticks)
        plt.setp(axes_comb[1].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout slightly for title
    plt.show() # Show the combined plot
else:
    print("\nSkipping combined visualization: No historical data or no predictions generated.")


print("\n--- Processing Complete ---")


# --- Visualization: Combined Plot showing Historical and Predicted Data ---
print("\n--- Generating Combined Visualization for Predicted Years ---")
if not yearly_avg_mpce.empty and not predictions_all_df.empty:
    fig_comb, axes_comb = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig_comb.suptitle(f'Historical MPCE and Predictions for {prediction_years_num[0]}-{prediction_years_num[-1]}', fontsize=16, y=1.0)

    # Ensure 'Year' in yearly_avg_mpce is numeric for plotting
    yearly_avg_mpce['Year'] = pd.to_numeric(yearly_avg_mpce['Year'], errors='coerce')
    yearly_avg_mpce.dropna(subset=['Year'], inplace=True)

    # --- Rural Combined Plot ---
    # Plot Historical Data
    axes_comb[0].plot(yearly_avg_mpce['Year'], yearly_avg_mpce['MPCE_Rural'], marker='o', linestyle='-', color='blue', label='Historical Avg Rural', alpha=0.7, markersize=6)

    # Plot Predicted Data for each method
    plot_markers = {
        'Interp_Linear': ('^', 'cyan', 'Interp Linear'),
        'Interp_Spline': ('v', 'lime', 'Interp Spline'),
        'Linear_Avg': ('x', 'orange', 'Linear Reg'),
        'Poly_Avg': ('+', 'magenta', 'Poly Reg'),
        'SVR_Avg': ('s', 'brown', 'SVR'),
        'GPR_Avg': ('d', 'gold', 'GPR')
    }
    for method, (marker, color, label) in plot_markers.items():
        col_name = f'{method}_Rural'
        # Check if the column exists and has non-NA values to plot
        if col_name in predictions_all_df.columns and predictions_all_df[col_name].notna().any():
            # Plot only the predicted points that are not NaN
            plot_data = predictions_all_df.dropna(subset=[col_name])
            # Ensure Year_Num is numeric for plotting
            plot_data['Year_Num'] = pd.to_numeric(plot_data['Year_Num'], errors='coerce')
            plot_data.dropna(subset=['Year_Num'], inplace=True)
            axes_comb[0].plot(plot_data['Year_Num'], plot_data[col_name], marker=marker, markersize=8, color=color, linestyle='None', label=f'{label} Pred.')

    axes_comb[0].set_title('Average Rural MPCE')
    axes_comb[0].set_ylabel('Average MPCE (Rs.)')
    axes_comb[0].legend(fontsize='small', loc='upper left', ncol=2) # Adjust legend position/columns
    axes_comb[0].grid(True, linestyle=':', alpha=0.6)

    # --- Urban Combined Plot ---
     # Plot Historical Data
    axes_comb[1].plot(yearly_avg_mpce['Year'], yearly_avg_mpce['MPCE_Urban'], marker='o', linestyle='-', color='green', label='Historical Avg Urban', alpha=0.7, markersize=6)

    # Plot Predicted Data for each method
    for method, (marker, color, label) in plot_markers.items():
        col_name = f'{method}_Urban'
        if col_name in predictions_all_df.columns and predictions_all_df[col_name].notna().any():
             # Plot only the predicted points that are not NaN
            plot_data = predictions_all_df.dropna(subset=[col_name])
             # Ensure Year_Num is numeric for plotting
            plot_data['Year_Num'] = pd.to_numeric(plot_data['Year_Num'], errors='coerce')
            plot_data.dropna(subset=['Year_Num'], inplace=True)
            axes_comb[1].plot(plot_data['Year_Num'], plot_data[col_name], marker=marker, markersize=8, color=color, linestyle='None', label=f'{label} Pred.')

    axes_comb[1].set_title('Average Urban MPCE')
    axes_comb[1].set_xlabel('Year')
    axes_comb[1].set_ylabel('Average MPCE (Rs.)')
    axes_comb[1].legend(fontsize='small', loc='upper left', ncol=2) # Adjust legend position/columns
    axes_comb[1].grid(True, linestyle=':', alpha=0.6)

    # Set x-axis ticks for combined plot - include historical and predicted years
    # Ensure all years used for ticks are numeric
    hist_years = pd.to_numeric(yearly_avg_mpce['Year'], errors='coerce').dropna().unique()
    pred_years = pd.to_numeric(predictions_all_df['Year_Num'], errors='coerce').dropna().unique()
    all_plot_years = sorted(list(set(hist_years) | set(pred_years)))

    if len(all_plot_years)>15 :
       step_comb = max(1, len(all_plot_years)//12) # Ensure step is at least 1
       # Select ticks ensuring they are integers
       selected_ticks = [int(yr) for yr in all_plot_years[::step_comb]]
       axes_comb[1].set_xticks(selected_ticks)
       plt.setp(axes_comb[1].get_xticklabels(), rotation=45, ha="right")
    elif len(all_plot_years)>0:
       # Ensure ticks are integers
       selected_ticks = [int(yr) for yr in all_plot_years]
       axes_comb[1].set_xticks(selected_ticks)
       plt.setp(axes_comb[1].get_xticklabels(), rotation=45, ha="right")


    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout slightly for title
    plt.show()
else:
    print("\nSkipping combined visualization: No historical data or no predictions generated.")


print("\n--- Processing Complete ---")
