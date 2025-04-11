import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gainAdjustment(df, min_gain=-48, max_gain=24):
    """
    Apply logarithmic gain correction to tracks that have very low gain.
    This function focuses on the gain values only, without relying on track names.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing mixing parameters, with 'Gain dB' column
    min_gain : float
        Minimum gain value (-48 dB)
    max_gain : float
        Maximum gain value (24 dB)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with corrected gain values
    """
    # Create a copy to avoid modifying the original
    corrected_df = df.copy()
    
    # Define correction parameters
    gain_range = max_gain - min_gain
    threshold = min_gain + gain_range * 0.5  # Apply correction below this threshold
    
    # Logarithmic correction function parameters
    # The closer to min_gain, the stronger the correction
    correction_strength = 36  # Maximum dB boost for completely silenced tracks
    
    # Process each track
    for track_name in corrected_df.index:
        current_gain = corrected_df.loc[track_name, 'Gain dB']
        
        # Apply correction only to tracks below threshold
        if current_gain < threshold:
            # Calculate how close we are to min_gain (0 = at threshold, 1 = at min_gain)
            silence_factor = (threshold - current_gain) / (threshold - min_gain)
            
            # Exponentially stronger correction for very low gains
            # This creates a curve that gives moderate correction for somewhat low gains
            # but much stronger correction for near-silent tracks
            correction = correction_strength * (silence_factor ** 2)
            
            # Apply the correction
            corrected_df.loc[track_name, 'Gain dB'] = current_gain + correction
            
            print(f"Track: {track_name}, Original: {current_gain:.1f} dB, Corrected: {current_gain + correction:.1f} dB")
    

    return corrected_df

# def plotGain(min_gain=-48, max_gain=24, threshold_percent=0.5, correction_strength=36, 
#                             output_path="./results/gain_correction_curve.png"):
#     """
#     Visualize the gain correction curve for the full range of possible gain values.
    
#     Parameters:
#     -----------
#     min_gain : float
#         Minimum gain value (-48 dB)
#     max_gain : float
#         Maximum gain value (24 dB)
#     threshold_percent : float
#         Percentage of gain range for correction threshold (0.0 to 1.0)
#     correction_strength : float
#         Maximum correction in dB applied to the lowest gain values
#     output_path : str
#         Path to save the visualization
        
#     """
#     # Calculate the threshold
#     gain_range = max_gain - min_gain
#     threshold = min_gain + gain_range * threshold_percent
    
#     # Create input gain values ranging from min_gain to max_gain
#     original_gains = np.linspace(min_gain, max_gain, 1000)
#     corrected_gains = np.copy(original_gains)
    
#     # Apply correction to values below threshold
#     mask = original_gains < threshold
#     silence_factor = (threshold - original_gains[mask]) / (threshold - min_gain)
#     correction = correction_strength * (silence_factor ** 2)
#     corrected_gains[mask] = original_gains[mask] + correction
    
#     # Create the figure
#     plt.figure(figsize=(12, 8))
    
#     # Plot the original vs. corrected gain
#     plt.plot(original_gains, corrected_gains, 'b-', linewidth=2.5, label='Corrected Gain')
#     plt.plot(original_gains, original_gains, 'k--', linewidth=1.5, label='Original Gain (no correction)')
    
#     # Plot the correction amount
#     correction_values = np.zeros_like(original_gains)
#     correction_values[mask] = correction
#     plt.plot(original_gains, correction_values, 'r-', linewidth=2, label='Correction Amount (dB)')
    
#     # Add reference lines
#     plt.axvline(x=threshold, color='g', linestyle='--', linewidth=1.5, 
#                label=f'Threshold ({threshold:.1f} dB)')
#     plt.axvline(x=min_gain, color='gray', linestyle=':', linewidth=1)
#     plt.axvline(x=max_gain, color='gray', linestyle=':', linewidth=1)
    
#     # Set axis labels and title
#     plt.xlabel('Original Gain (dB)', fontsize=12)
#     plt.ylabel('Gain (dB)', fontsize=12)
#     plt.title('Logarithmic Gain Correction Curve', fontsize=14)
    
#     # Add grid and legend
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=12)
    
#     # Annotate key points
#     plt.annotate(f'Min Gain: {min_gain} dB', xy=(min_gain, min_gain), xytext=(min_gain+5, min_gain-10),
#                 arrowprops=dict(arrowstyle='->'))
#     plt.annotate(f'Max Gain: {max_gain} dB', xy=(max_gain, max_gain), xytext=(max_gain-15, max_gain+10),
#                 arrowprops=dict(arrowstyle='->'))
#     plt.annotate(f'Max Correction: {correction_strength} dB', 
#                 xy=(min_gain, min_gain+correction_strength), 
#                 xytext=(min_gain+10, min_gain+correction_strength+5),
#                 arrowprops=dict(arrowstyle='->'))
    
#     # Set limits with some padding
#     plt.xlim(min_gain - 5, max_gain + 5)
#     plt.ylim(min_gain - 5, max_gain + 5)
    
#     # Save the figure
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     print(f"Visualization saved to {output_path}")