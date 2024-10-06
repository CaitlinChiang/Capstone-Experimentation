import parselmouth
from parselmouth.praat import call

# Function to extract intervals and texts from TextGrid file
def read_textgrid(file_path):
    # Load the TextGrid file using parselmouth
    textgrid = parselmouth.read(file_path)
    
    # Extract tiers (layers of annotations)
    tiers = textgrid.tiers
    
    for tier in tiers:
        # Get tier name
        tier_name = call(tier, "Get name")
        print(f"Tier Name: {tier_name}")
        
        # Iterate through intervals in the tier
        for interval_index in range(call(tier, "Get number of intervals")):
            # Get interval boundaries and the associated text
            start_time = call(tier, "Get start time", interval_index + 1)
            end_time = call(tier, "Get end time", interval_index + 1)
            text = call(tier, "Get text", interval_index + 1)
            
            # Print interval information
            print(f"Interval {interval_index + 1}: {start_time} - {end_time}, Text: {text}")

# Path to your .TextGrid file
file_path = "data/textgrid/app_4001_6001_phnd_deb-1.TextGrid"

# Call the function to read and print textgrid contents
read_textgrid(file_path)
