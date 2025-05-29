import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Maximum number of records to maintain
MAX_RECORDS = 1000

# Generate timestamps between November 1st, 2024 and today
start_date = datetime(2024, 11, 1)
end_date = datetime.now()

# Calculate the total number of days between start and end date
total_days = (end_date - start_date).days

def generate_new_records(n_records):
    # Generate random timestamps within this range
    timestamps = [start_date + timedelta(days=random.randint(0, total_days), 
                                      hours=random.randint(0, 23),
                                      minutes=random.randint(0, 59),
                                      seconds=random.randint(0, 59)) 
                 for _ in range(n_records)]
    timestamps.sort()

    # Generate realistic medical data
    data = {
        'time': [ts.strftime('%d/%m/%Y (%I:%M:%S %p)') for ts in timestamps],
        'age': np.random.randint(20, 80, n_records),
        'bloodPressure': np.random.randint(60, 180, n_records),
        'sugar': np.random.randint(0, 5, n_records),
        'pusCell': np.random.randint(0, 2, n_records),
        'pusCellClumps': np.random.randint(0, 2, n_records),
        'sodium': np.random.randint(0, 150, n_records),
        'hemoglobin': np.random.uniform(8, 25, n_records).round(1),
        'hypertension': np.random.randint(0, 2, n_records),
        'diabetesMelitus': np.random.randint(0, 2, n_records)
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Generate results based on medical rules
    def determine_result(row):
        # Define risk factors
        risk_factors = 0
        
        # Age risk (over 50)
        if row['age'] > 50:
            risk_factors += 1
        
        # Blood pressure risk (high or low)
        if row['bloodPressure'] < 70 or row['bloodPressure'] > 140:
            risk_factors += 1
        
        # Sugar level risk
        if row['sugar'] > 2:
            risk_factors += 1
        
        # Pus cell risk
        if row['pusCell'] == 1 or row['pusCellClumps'] == 1:
            risk_factors += 1
        
        # Sodium risk
        if row['sodium'] < 10 or row['sodium'] > 140:
            risk_factors += 1
        
        # Hemoglobin risk
        if row['hemoglobin'] < 12 or row['hemoglobin'] > 20:
            risk_factors += 1
        
        # Existing conditions risk
        if row['hypertension'] == 1 or row['diabetesMelitus'] == 1:
            risk_factors += 1
        
        # Determine result based on risk factors
        if risk_factors >= 3:
            return 1  # Kidney Disease
        else:
            return 0  # Healthy

    # Add results column
    df['result'] = df.apply(determine_result, axis=1)
    return df

def maintain_record_limit():
    # Check if records file exists
    if os.path.exists('dataset/records.csv'):
        # Read existing records
        existing_df = pd.read_csv('dataset/records.csv')
        
        # Convert time column to datetime for sorting
        existing_df['time'] = pd.to_datetime(existing_df['time'], format='%d/%m/%Y (%I:%M:%S %p)')
        
        # Sort by time in descending order (newest first)
        existing_df = existing_df.sort_values('time', ascending=False)
        
        # Keep only the most recent MAX_RECORDS - 100 records
        existing_df = existing_df.head(MAX_RECORDS - 1000)
        
        # Generate 100 new records
        new_records = generate_new_records(1000)
        
        # Convert new records time to datetime for sorting
        new_records['time'] = pd.to_datetime(new_records['time'], format='%d/%m/%Y (%I:%M:%S %p)')
        
        # Combine existing and new records
        combined_df = pd.concat([existing_df, new_records], ignore_index=True)
        
        # Sort by time in descending order
        combined_df = combined_df.sort_values('time', ascending=False)
        
        # Keep only the most recent MAX_RECORDS
        combined_df = combined_df.head(MAX_RECORDS)
        
        # Convert time back to string format
        combined_df['time'] = combined_df['time'].dt.strftime('%d/%m/%Y (%I:%M:%S %p)')
        
        # Save to CSV
        combined_df.to_csv('dataset/records.csv', index=False)
        
        print("\nUpdated Dataset Statistics:")
        print("-" * 30)
        print(f"Total Records: {len(combined_df)}")
        print(f"Date Range: {combined_df['time'].min()} to {combined_df['time'].max()}")
        print(f"Healthy Patients: {len(combined_df[combined_df['result'] == 0])} ({len(combined_df[combined_df['result'] == 0])/len(combined_df)*100:.1f}%)")
        print(f"Patients with Kidney Disease: {len(combined_df[combined_df['result'] == 1])} ({len(combined_df[combined_df['result'] == 1])/len(combined_df)*100:.1f}%)")
        
        print("\nSample of Most Recent Records:")
        print("-" * 30)
        print(combined_df.head())
    else:
        # If no existing records, generate MAX_RECORDS new ones
        df = generate_new_records(MAX_RECORDS)
        df.to_csv('dataset/records.csv', index=False)
        
        print("\nGenerated New Dataset Statistics:")
        print("-" * 30)
        print(f"Total Records: {len(df)}")
        print(f"Date Range: {df['time'].min()} to {df['time'].max()}")
        print(f"Healthy Patients: {len(df[df['result'] == 0])} ({len(df[df['result'] == 0])/len(df)*100:.1f}%)")
        print(f"Patients with Kidney Disease: {len(df[df['result'] == 1])} ({len(df[df['result'] == 1])/len(df)*100:.1f}%)")
        
        print("\nSample of Records:")
        print("-" * 30)
        print(df.head())

if __name__ == "__main__":
    maintain_record_limit() 