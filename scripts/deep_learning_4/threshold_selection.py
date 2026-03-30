import numpy as np
from scipy.stats import beta

class ThresholdSelector:
    # Initialize with the bandwidth in Hz (default 100 MHz as per simulator configuration)
    def __init__(self, bandwidth_hz=100e6):
        self.bandwidth_hz = bandwidth_hz

    # Converts SINR (dB) to an equivalent throughput (Mbps) using the Shannon formula.
    # Equation: R = BW * log2(1 + SINR_linear), R is the throughput in bps, BW is the bandwidth in Hz, and SINR_linear is the linear scale of SINR.
    def shannon_throughput(self, sinr_db):
        sinr_linear = 10 ** (sinr_db / 10)
        throughput_bps = self.bandwidth_hz * np.log2(1 + sinr_linear)
        return throughput_bps / 1e6  # Return in Mbps

    # Inverse Shannon formula: Finds the minimum SINR (dB) required to hit a target Throughput.
    def required_sinr_for_throughput(self, target_tput_mbps):
        target_bps = target_tput_mbps * 1e6
        sinr_linear = (2 ** (target_bps / self.bandwidth_hz)) - 1
        
        if sinr_linear <= 0:
            return -np.inf # Impossible to calculate log10 of a negative number
            
        return 10 * np.log10(sinr_linear)

    # AR Constraint Function to find the margin based on historical residuals and a reliability threshold (epsilon).
    # Epsilon is the maximum allowed average switching probability, which translates to a percentile of the error distribution.
    # NOTE: Epsilon set at 0.7 by default from proposal
    def get_ar_margin(self, residuals, epsilon=0.7):
        percentile_target = (1.0 - epsilon) * 100
        return np.percentile(residuals, percentile_target)

    # PCR Constraint Function to find a stricter margin based on the Beta distribution of residuals and a confidence level.
    # NOTE: xi set at 0.05 and confidence at 0.95 by default from proposal
    def get_pcr_margin(self, residuals, xi=0.05, confidence=0.95):
        n = len(residuals)
        sorted_res = np.sort(residuals)
        
        # Find the order statistic index 'k' using the Beta distribution CDF.
        # NOTE: This is the standard non-parametric mathematical approach for PCR from the research paper.
        k = 1
        while k < n:
            prob = 1.0 - beta.cdf(1 - xi, k, n - k + 1)
            if prob >= confidence:
                break
            k += 1
            
        # Ensure k does not exceed the array bounds
        k = min(k, n - 1)
        
        # Return the specific error value at that strict index
        return sorted_res[k - 1]

    # Applies the AR or PCR margin to the deep learning prediction.
    def get_corrected_sinr(self, predicted_sinr_db, margin_db):
        return predicted_sinr_db + margin_db