import pandas as pd

def analyze_similarity_csv(file_path):
    try:
        # Read the CSV file
        # If your file has NO headers at all, change this to pd.read_csv(file_path, header=None)
        df = pd.read_csv(file_path)
        
        # Select the 3rd column (index 2)
        target_column = df.columns[2]
        values = df[target_column]
        
        # Calculate statistics
        min_val = values.min()
        max_val = values.max()
        avg_val = values.mean()
        
        print(f"📊 Analysis for: {file_path}")
        print(f"   Absolute Range : {min_val:.4f} to {max_val:.4f}")
        print(f"   Average Score  : {avg_val:.4f}")
        print("-" * 40)
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file '{file_path}'")
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")

# Run the analysis on both of your files
if __name__ == "__main__":
    analyze_similarity_csv("ML_data/positive_similarity.csv")
    analyze_similarity_csv("ML_data/negative_similarity.csv")