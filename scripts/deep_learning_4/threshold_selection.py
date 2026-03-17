import numpy as np
from scipy.stats import beta

class ThresholdSelector:
    """
    Implements Step 9 of the FYP Proposal: Threshold Selection using Reliability Constraints.
    """
    def __init__(self, bandwidth_hz=100e6):
        # Default bandwidth is 100 MHz based on the simulator configuration
        self.bandwidth_hz = bandwidth_hz

    # ==========================================
    # STEP 9.4: SHANNON CAPACITY CONVERSIONS
    # ==========================================
    def shannon_throughput(self, sinr_db):
        """
        Converts SINR (dB) to an equivalent throughput (Mbps) using the Shannon formula.
        Equation: R = BW * log2(1 + SINR_linear)
        """
        sinr_linear = 10 ** (sinr_db / 10)
        throughput_bps = self.bandwidth_hz * np.log2(1 + sinr_linear)
        return throughput_bps / 1e6  # Return in Mbps

    def required_sinr_for_throughput(self, target_tput_mbps):
        """
        Inverse Shannon formula: Finds the minimum SINR (dB) required to hit a target Throughput.
        """
        target_bps = target_tput_mbps * 1e6
        sinr_linear = (2 ** (target_bps / self.bandwidth_hz)) - 1
        
        if sinr_linear <= 0:
            return -np.inf # Impossible to calculate log10 of a negative number
            
        return 10 * np.log10(sinr_linear)

    # ==========================================
    # STEP 9.2 & 9.3: RELIABILITY CONSTRAINTS
    # ==========================================
    def get_ar_margin(self, residuals, epsilon=0.7):
        """
        Step 9.2: Average Reliability (AR) Constraint.
        Ensures the average switching probability <= epsilon.
        
        If epsilon is 0.7 (70% reliability), we look at the 30th percentile 
        of our historical errors to find a safe "padding" to add to our predictions.
        """
        percentile_target = (1.0 - epsilon) * 100
        return np.percentile(residuals, percentile_target)

    def get_pcr_margin(self, residuals, xi=0.05, confidence=0.95):
        """
        Step 9.3: Probably Correct Reliability (PCR) Constraint.
        Uses Beta-distributed order statistics to find a stricter, worst-case bound.
        
        This calculates: "I am 95% confident that my error will not be worse than X."
        """
        n = len(residuals)
        sorted_res = np.sort(residuals)
        
        # We find the order statistic index 'k' using the Beta distribution CDF.
        # This is the standard non-parametric mathematical approach for PCR.
        k = 1
        while k < n:
            # Probability that the k-th smallest error covers our requirement
            prob = 1.0 - beta.cdf(1 - xi, k, n - k + 1)
            if prob >= confidence:
                break
            k += 1
            
        # Ensure k doesn't exceed the array bounds
        k = min(k, n - 1)
        
        # Return the specific error value at that strict index
        return sorted_res[k - 1]

    # ==========================================
    # FINAL CORRECTION APPLICATION
    # ==========================================
    def get_corrected_sinr(self, predicted_sinr_db, margin_db):
        """
        Applies the AR or PCR margin to the deep learning prediction.
        Because 'True SINR = Predicted + Error', adding the negative margin 
        pulls the prediction down to a safe, conservative level.
        """
        return predicted_sinr_db + margin_db