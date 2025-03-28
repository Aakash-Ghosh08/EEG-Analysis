import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import skew, kurtosis
import mne
from mne.preprocessing import ICA
import antropy as ant
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

def load_and_preprocess_eeg(file_path, fs=250):
    """Load and preprocess EEG data."""
    try:
        # Load the data
        df = pd.read_csv(file_path, delimiter="\t")
        
        # Identify the EEG data column (potentially the first numeric column)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # If no specific column is specified, use the first numeric column
        eeg_col = numeric_columns[0]
        print(f"Using column '{eeg_col}' as EEG data")
        
        # Extract timestamp and convert to datetime
        timestamp_col = df.columns[-2]
        df["Timestamp"] = pd.to_datetime(df[timestamp_col], unit="s")
        
        # Handle missing values
        df.dropna(subset=[eeg_col], inplace=True)
        
        # Convert to numeric and handle potential parsing errors
        df[eeg_col] = pd.to_numeric(df[eeg_col], errors='coerce')
        
        # Normalize the signal (optional, but can help with scaling)
        df["Normalized_EEG"] = (df[eeg_col] - df[eeg_col].mean()) / df[eeg_col].std()
        
        # Detrend the data
        df["Detrended_EEG"] = signal_detrend(df["Normalized_EEG"])
        
        # Apply filters
        df["Filtered_EEG"] = apply_all_filters(df["Detrended_EEG"], fs)
        
        # Detect and remove artifacts (simplified approach)
        df["Clean_EEG"] = remove_artifacts(df["Filtered_EEG"])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def signal_detrend(data):
    """Remove linear trend from signal."""
    return data - np.polyval(np.polyfit(np.arange(len(data)), data, 1), np.arange(len(data)))

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass filter to the data."""
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def notch_filter(data, freq, fs, q=30):
    """Apply notch filter to remove line noise."""
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, data)

def apply_all_filters(data, fs):
    """Apply all necessary filters to the data."""
    # Bandpass filter (0.5-50 Hz)
    filtered = bandpass_filter(data, 0.5, 50, fs, order=4)
    
    # Notch filter for line noise (50 Hz)
    filtered = notch_filter(filtered, 50, fs)
    
    # Additional notch filter for US line noise (60 Hz) if needed
    # filtered = notch_filter(filtered, 60, fs)
    
    return filtered

def remove_artifacts(data, threshold=3.5):
    """Simple artifact removal based on amplitude threshold."""
    # Z-score based thresholding
    z_scores = (data - data.mean()) / data.std()
    clean_data = data.copy()
    
    # Replace extreme values with median instead of NaN
    median = np.median(data)
    clean_data[abs(z_scores) > threshold] = median
    
    return clean_data

def compute_psd(data, fs, nperseg=None):
    """Compute power spectral density using Welch's method."""
    if nperseg is None:
        nperseg = min(256, len(data)//4)
    
    # Apply Hanning window and compute PSD
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, window='hann')
    return freqs, psd

def compute_band_powers(data, fs):
    """Compute absolute and relative band powers."""
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (12, 30),
        "Gamma": (30, 50)
    }
    
    # Compute PSD
    freqs, psd = compute_psd(data, fs)
    
    # Calculate absolute band powers
    abs_band_powers = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        abs_band_powers[band] = np.trapz(psd[idx], freqs[idx])
    
    # Calculate total power
    total_power = sum(abs_band_powers.values())
    
    # Calculate relative band powers
    rel_band_powers = {band: max(0, power/total_power) for band, power in abs_band_powers.items()}
    
    return abs_band_powers, rel_band_powers

def compute_features(data):
    """Compute various time-domain and complexity features."""
    features = {
        # Statistical features
        "Mean": np.mean(data),
        "Std_Dev": np.std(data),
        "Variance": np.var(data),
        "Skewness": skew(data),
        "Kurtosis": kurtosis(data),
        
        # Range features
        "Range": np.max(data) - np.min(data),
        "Interquartile_Range": np.percentile(data, 75) - np.percentile(data, 25),
        
        # Complexity measures
        "Sample_Entropy": ant.sample_entropy(data),
        "Perm_Entropy": ant.perm_entropy(data, normalize=True),
        "Hjorth_Mobility": ant.hjorth_params(data)[0],
        "Hjorth_Complexity": ant.hjorth_params(data)[1],
    }
    
    return features

def plot_spectrogram(data, fs, title="Spectrogram"):
    """Plot spectrogram of EEG data."""
    plt.figure(figsize=(10, 6))
    
    # Compute and plot spectrogram
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=256, noverlap=128)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.ylim(0, 50)  # Limit frequency display to 0-50 Hz
    plt.tight_layout()
    
    return plt.gcf()

def plot_signals_and_psd(df, fs):
    """Plot signals and power spectral density."""
    # Create figure with GridSpec for better layout
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # Time domain signals plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["Timestamp"], df[df.columns[1]], label="Raw EEG", alpha=0.5, color='gray')
    ax1.plot(df["Timestamp"], df["Filtered_EEG"], label="Filtered EEG", color='blue')
    ax1.plot(df["Timestamp"], df["Clean_EEG"], label="Clean EEG", color='green')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude (µV)")
    ax1.legend()
    ax1.set_title("EEG Signal Comparison")
    
    # PSD plots
    ax2 = fig.add_subplot(gs[1, 0])
    freqs, psd_raw = compute_psd(df[df.columns[1]], fs)
    ax2.semilogy(freqs, psd_raw, label="Raw EEG")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power Spectral Density (µV²/Hz)")
    ax2.set_title("Raw EEG - PSD")
    ax2.set_xlim(0, 50)
    ax2.grid(True)
    
    ax3 = fig.add_subplot(gs[1, 1])
    freqs, psd_clean = compute_psd(df["Clean_EEG"], fs)
    ax3.semilogy(freqs, psd_clean, label="Clean EEG", color='green')
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Power Spectral Density (µV²/Hz)")
    ax3.set_title("Clean EEG - PSD")
    ax3.set_xlim(0, 50)
    ax3.grid(True)
    
    # Band power comparison plot
    ax4 = fig.add_subplot(gs[2, :])
    
    # Compute relative band powers
    _, rel_band_powers = compute_band_powers(df["Clean_EEG"], fs)
    bands = list(rel_band_powers.keys())
    powers = list(rel_band_powers.values())
    
    ax4.bar(bands, powers, color=['blue', 'green', 'red', 'purple', 'orange'])
    ax4.set_ylabel("Relative Power")
    ax4.set_title("EEG Frequency Bands - Relative Power")
    
    plt.tight_layout()
    return fig

def analyze_eeg(file_path, fs=250):
    """Main function to analyze EEG data."""
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_eeg(file_path, fs)
    if df is None:
        return
    
    # Extract features
    print("\nExtracting features...")
    features = compute_features(df["Clean_EEG"])
    print("Time-domain and complexity features:")
    for feature, value in features.items():
        print(f"{feature}: {value:.4f}")
    
    # Compute band powers
    print("\nComputing frequency band powers...")
    abs_band_powers, rel_band_powers = compute_band_powers(df["Clean_EEG"], fs)
    
    print("\nAbsolute band powers:")
    for band, power in abs_band_powers.items():
        print(f"{band}: {power:.4f}")
        
    print("\nRelative band powers:")
    for band, power in rel_band_powers.items():
        print(f"{band}: {power:.4f}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_signals_and_psd(df, fs)
    plt.savefig("eeg_analysis_summary.png", dpi=300)
    plt.show()
    
    # Generate spectrogram
    plot_spectrogram(df["Clean_EEG"], fs, title="EEG Spectrogram")
    plt.savefig("eeg_spectrogram.png", dpi=300)
    plt.show()
    
    return df, features, abs_band_powers, rel_band_powers

if __name__ == "__main__":
    file_path = "C:/Users/aakas/Documents/OpenBCI_GUI/Recordings/OpenBCISession_2025-03-15_18-10-25 P9/BrainFlow-RAW_2025-03-15_18-10-25 P9_0.csv"
    fs = 250  # Sampling frequency in Hz
    
    print("Starting EEG analysis...")
    analyze_eeg(file_path, fs)
    print("Analysis complete!")