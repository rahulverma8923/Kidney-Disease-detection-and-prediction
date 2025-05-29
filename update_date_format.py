import pandas as pd
import re
from datetime import datetime

# Read the records CSV file
df = pd.read_csv('dataset/records.csv')

# Function to convert date format from mm/dd/yyyy to dd/mm/yyyy and time to AM/PM format
def convert_date_format(date_string):
    # Extract the date part and time part
    parts = date_string.split(' ')
    date_part = parts[0]
    
    # Handle the time portion if it exists
    time_part = ""
    if len(parts) > 1:
        time_part = parts[1]
    
    # Convert date format if needed
    if re.match(r'^\d{2}/\d{2}/\d{4}$', date_part):
        date_components = date_part.split('/')
        # Check if first component looks like a month (1-12)
        if int(date_components[0]) <= 12 and int(date_components[0]) != int(date_components[1]):
            # Swap month and day
            day = date_components[1]
            month = date_components[0]
            year = date_components[2]
            date_part = f"{day}/{month}/{year}"
    
    # Convert time format if it's in the format (HH:MM:SS)
    if time_part and time_part.startswith('(') and ':' in time_part:
        # Extract just the time numbers
        time_str = time_part.strip('()')
        
        # Check if it doesn't already have AM/PM
        if not ('AM' in time_str or 'PM' in time_str):
            try:
                # Parse the time
                time_obj = datetime.strptime(time_str, "%H:%M:%S")
                # Convert to 12-hour format with AM/PM
                time_12h = time_obj.strftime("%I:%M:%S %p")
                time_part = f"({time_12h})"
            except ValueError:
                # If there's any error in parsing, keep the original
                pass
    
    # Combine date and time parts
    if time_part:
        return date_part + " " + time_part
    else:
        return date_part

# Apply the conversion to the time column
df['time'] = df['time'].apply(convert_date_format)

# Save the updated records
df.to_csv('dataset/records.csv', index=False)

print("Records updated successfully. All dates are now in dd/mm/yyyy format and times in AM/PM format.") 