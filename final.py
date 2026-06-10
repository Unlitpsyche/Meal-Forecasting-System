# STEP 1: Import necessary libraries
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import re

# STEP 2: Load and combine all data files
print("Starting data loading and preprocessing...")

base_path = r"C:\Users\Dell\OneDrive\Desktop\Deku\files"

# This list will store a DataFrame for each day's data
all_data_list = []

# Loop through all 14 data files (data1.csv to data14.csv)
for i in range(1, 15):
    file_name = f"data{i}.csv"
    file_path = os.path.join(base_path, file_name)

    # Check if the file exists before trying to read it
    if os.path.exists(file_path):
        print(f"Processing file: {file_name}")

        try:
            # First, read the file to find the date and the actual data start row.
            # We read the first few rows to find the date string and the header.
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            date_str = None
            skip_rows = 0
            for row_num, line in enumerate(lines):
                # Use a regular expression to find the full date string
                date_match = re.search(r'Date\s*:\s*(.*)', line)
                if date_match:
                    # Extract the first part of the captured group, before any extra commas
                    extracted_date_part = date_match.group(1).strip().replace('"', '').split(',,')[0].strip()
                    # Now, try to parse this part as the date
                    try:
                        date_obj = pd.to_datetime(extracted_date_part, format='%A, %d %B %Y')
                        date_str = extracted_date_part
                    except ValueError:
                        # If parsing fails with the current method, try a more flexible approach
                        try:
                            date_obj = pd.to_datetime(extracted_date_part, format='%d %B %Y')
                            date_str = extracted_date_part
                        except ValueError:
                            # If all else fails, log the error and skip this file's date parsing
                            print(f"  Could not parse date: '{extracted_date_part}'. Skipping date parsing for this file.")
                            date_str = None
                            date_obj = None

                # Find the line containing the header and determine how many rows to skip
                if 'Meals Served' in line:
                    skip_rows = row_num + 1
                    break
            
            # If we didn't find the date or header, skip this file.
            if not date_str or not skip_rows:
                print(f"  Could not find date or header in {file_name}. Skipping.")
                continue

            # Now, read the main data, skipping the determined number of header rows.
            df = pd.read_csv(file_path, skiprows=skip_rows - 1)
            
            # Drop the last row which contains 'TOTAL', if it exists.
            # We convert the cell content to a string to avoid the 'float is not iterable' error.
            if not df.empty and 'TOTAL' in str(df.iloc[-1, 0]):
                df = df.iloc[:-1]

            # The 'Meals Served' and 'Meals not Served' columns may contain percentages or be a number.
            # We need a robust way to extract the numerical value.
            def extract_meals_value(s):
                if pd.isna(s):
                    return 0
                # If the value is already a number (float or int), return it directly.
                if isinstance(s, (int, float)):
                    return int(s)
                # If the value is a string, use a regex to find the first sequence of digits.
                if isinstance(s, str):
                    match = re.search(r'(\d+)', s)
                    if match:
                        return int(match.group(1))
                return 0

            # This handles cases where the column names might have leading/trailing spaces.
            df.columns = df.columns.str.strip()
            df['Meals Served Value'] = df['Meals Served'].apply(extract_meals_value)
            df['Meals not Served Value'] = df['Meals not Served'].apply(extract_meals_value)

            # Calculate the total number of meals required for the day
            total_meals_required = df['Meals Served Value'].sum() + df['Meals not Served Value'].sum()

            # Create a temporary DataFrame for this day with a 'ds' and 'y' column,
            temp_df = pd.DataFrame({
                'ds': [date_obj],
                'y': [total_meals_required]
            })

            all_data_list.append(temp_df)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    else:
        print(f"File not found: {file_path}")

# Concatenate all the individual dataframes into one master dataframe
if not all_data_list:
    print("No data files were processed. Please check the file paths and names.")
    exit()

master_df = pd.concat(all_data_list, ignore_index=True)

# Prophet requires the 'ds' column to be a datetime type.
master_df['ds'] = pd.to_datetime(master_df['ds'])

# STEP 3: Load and combine 'why' files as extra regressors
print("\nLoading and combining 'why' files for extra regressors...")
why_data_list = []
# Loop through all 14 why files (why1.csv to why14.csv)
for i in range(1, 15):
    file_name = f"why{i}.csv"
    file_path = os.path.join(base_path, file_name)

    if os.path.exists(file_path):
        print(f"Processing why file: {file_name}")
        try:
            # Read the 'why' file to find the date and header.
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            date_str = None
            skip_rows = 0
            for row_num, line in enumerate(lines):
                date_match = re.search(r'Date\s*:\s*(.*)', line)
                if date_match:
                    extracted_date_part = date_match.group(1).strip().replace('"', '').split(',,')[0].strip()
                    try:
                        date_obj = pd.to_datetime(extracted_date_part, format='%A, %d %B %Y')
                        date_str = extracted_date_part
                    except ValueError:
                        try:
                            date_obj = pd.to_datetime(extracted_date_part, format='%d %B %Y')
                            date_str = extracted_date_part
                        except ValueError:
                            date_str = None
                            date_obj = None

                if 'Meals not served' in line and 'Holidays' in line:
                    skip_rows = row_num + 1
                    break
            
            if not date_str or not skip_rows:
                print(f"  Could not find date or header in {file_name}. Skipping.")
                continue

            # Read the data and ensure columns are handled correctly
            df_why = pd.read_csv(file_path, skiprows=skip_rows - 1)
            
            if not df_why.empty and 'TOTAL' in str(df_why.iloc[-1, 0]):
                df_why = df_why.iloc[:-1]

            # The relevant columns are a fixed set for each file
            regressor_cols = ['No Ration', 'Funds', 'No cook', 'Food not arrived', 'Holidays', 'Others']
            
            # Sum the values for each regressor column
            regressor_data = {col: df_why[col].sum() for col in regressor_cols}
            regressor_data['ds'] = date_obj
            
            why_data_list.append(pd.DataFrame([regressor_data]))

        except Exception as e:
            print(f"Error processing why file {file_name}: {e}")
    else:
        print(f"Why file not found: {file_path}")

if not why_data_list:
    print("No 'why' files were processed. Proceeding with the base model.")
    final_df = master_df.copy()
    use_regressors = False
else:
    why_df = pd.concat(why_data_list, ignore_index=True)
    why_df['ds'] = pd.to_datetime(why_df['ds'])
    
    # Merge the two datasets on the date column
    final_df = pd.merge(master_df, why_df, on='ds', how='left')
    # Fill any missing regressor values with 0
    final_df[regressor_cols] = final_df[regressor_cols].fillna(0)
    use_regressors = True

# Sort the dataframe by date
final_df.sort_values('ds', inplace=True)
print("\nCombined and prepared data for Prophet (with extra regressors):")
print(final_df.head())

# STEP 4: Initialize and train the Prophet model
print("\nInitializing and training the Prophet model...")

# Create a new Prophet model instance
model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True,
    weekly_seasonality=True
)

# Add the extra regressors to the model before fitting
if use_regressors:
    for col in regressor_cols:
        model.add_regressor(col)

# Fit the model to our historical data
model.fit(final_df)

# STEP 5: Make future predictions for the next month and next year
print("Making predictions for the next month and next year...")

# Create a future dataframe for the next month (30 days) and next year (365 days)
future = model.make_future_dataframe(periods=365, freq='D')

# Here, we assume the reasons for meals not being served in the future are zero
if use_regressors:
    for col in regressor_cols:
        future[col] = 0

# Generate forecasts for the future dates
forecast = model.predict(future)

# STEP 6: Display the results and visualize
print("\nForecast for the next month:")
# The 'yhat' column contains the predicted values
print(forecast[['ds', 'yhat']].tail(30))

print("\nForecast for the next year:")
print(forecast[['ds', 'yhat']].tail(365))

# Calculate the total number of meals for the next month and year
forecast_month_total = forecast.tail(30)['yhat'].sum()
forecast_year_total = forecast.tail(365)['yhat'].sum()

print(f"\nTotal meals required for the next month: {int(forecast_month_total):,}")
print(f"Total meals required for the next year: {int(forecast_year_total):,}")

# Plot the forecast for the entire period (historical + future)
fig1 = model.plot(forecast)
plt.title('Meals Required Forecast (Historical and Future)')
plt.xlabel('Date')
plt.ylabel('Meals Required')
plt.savefig(os.path.join(base_path, 'forecast_plot_with_reasons.png'))
print("Forecast plot saved as 'forecast_plot_with_reasons.png'")

# Plot the seasonal and regressor components
fig2 = model.plot_components(forecast)
plt.suptitle('Prophet Seasonal and Regressor Components', y=1.02)
plt.savefig(os.path.join(base_path, 'components_plot_with_reasons.png'))
print("Components plot saved as 'components_plot_with_reasons.png'")

plt.show()

print("\nTask complete!")
