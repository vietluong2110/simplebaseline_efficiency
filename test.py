import os

def parse_traffic_results_by_predlen(file_path):
    """
    Reads result_long_term_forecast(backup).txt line by line, looking for entries
    starting with 'Traffic_'. Splits the line by underscores, extracts the prediction
    length from the third component, then parses the next line's MSE and MAE.
    
    Returns a dictionary:
       results_by_pred = {
          pred_len_1: [(config_line, mse, mae), ...],
          pred_len_2: [(config_line, mse, mae), ...],
          ...
       }
    """
    results_by_pred = {}
    current_config_line = None

    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # If the line starts with 'Traffic_', we'll parse it
        if line.startswith("Traffic_"):
            current_config_line = line
            
            # Split by underscore to extract the third component as pred_len
            # Example: Traffic_96_192_128_256_...
            parts = line.split("_")
            if len(parts) > 2:
                try:
                    pred_len = int(parts[2])  # the 3rd component is the prediction length
                except ValueError:
                    # If we can't parse as integer, skip
                    current_config_line = None
                    i += 1
                    continue
            else:
                current_config_line = None
                i += 1
                continue
            
            # Move to the next line to look for 'mse:..., mae:...'
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip().lower()
                if next_line.startswith("mse:"):
                    # Example next_line: "mse:0.37686, mae:0.40288"
                    mse_str, mae_str = next_line.split(",")
                    mse_val = float(mse_str.split(":")[1].strip())
                    mae_val = float(mae_str.split(":")[1].strip())

                    # Store in dictionary
                    if pred_len not in results_by_pred:
                        results_by_pred[pred_len] = []
                    results_by_pred[pred_len].append((current_config_line, mse_val, mae_val))

                    i += 2  # Skip the next line (already processed)
                    continue
        
        i += 1  # Move forward

    return results_by_pred


def find_best_configs(results_by_pred):
    """
    For each prediction length, finds the best configuration by MSE and by MAE.
    Returns a dict of dicts:
        best_configs = {
            pred_len: {
               'best_mse': (config_line, mse_val, mae_val),
               'best_mae': (config_line, mse_val, mae_val)
            },
            ...
        }
    """
    best_configs = {}
    for pred_len, entries in results_by_pred.items():
        if not entries:
            continue
        
        # Sort or just min() by MSE
        best_by_mse = min(entries, key=lambda x: x[1])  # x[1] = mse
        best_by_mae = min(entries, key=lambda x: x[2])  # x[2] = mae

        best_configs[pred_len] = {
            'best_mse': best_by_mse,
            'best_mae': best_by_mae
        }
    return best_configs


def main():
    file_path = "result_long_term_forecast(backup).txt"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Parse the file
    results_by_pred = parse_traffic_results_by_predlen(file_path)
    # Identify the best config for each prediction length
    best_configs = find_best_configs(results_by_pred)

    # Print results for each relevant prediction length
    # e.g., [96, 192, 336, 720] – adjust as needed
    pred_lengths = [96, 192, 336, 720]
    
    print("Best configurations by prediction length:")
    for pl in pred_lengths:
        if pl not in best_configs:
            print(f"\nPrediction length {pl}: No entries found.")
            continue
        
        best_mse_line, best_mse_val, best_mse_mae = best_configs[pl]['best_mse']
        best_mae_line, best_mae_mse, best_mae_val = best_configs[pl]['best_mae']
        
        print(f"\n=== Prediction length: {pl} ===")
        print(f"Best by MSE:\n  Config: {best_mse_line}\n  MSE: {best_mse_val:.6f}, MAE: {best_mse_mae:.6f}")
        print(f"Best by MAE:\n  Config: {best_mae_line}\n  MSE: {best_mae_mse:.6f}, MAE: {best_mae_val:.6f}")


if __name__ == "__main__":
    main()
