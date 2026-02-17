"""
Statistical Tests for Randomness of Student Selection

This script performs multiple statistical tests to determine whether
the teacher's student selection process is random or follows patterns.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, chisquare
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class RandomnessTestSuite:
    """Statistical tests for randomness in student selection."""
    
    def __init__(self, binary_vectors: np.ndarray, days: np.ndarray, total_students: int = 55):
        """
        Initialize the test suite.
        
        Args:
            binary_vectors: Binary matrix (n_samples, n_students)
            days: Array of day numbers
            total_students: Total number of students
        """
        self.binary_vectors = binary_vectors
        self.days = days
        self.total_students = total_students
        self.n_samples = len(binary_vectors)
        
        # Calculate selection frequencies
        self.selection_counts = binary_vectors.sum(axis=0)
        self.total_selections = self.selection_counts.sum()
        
    def chi_square_test_uniform(self) -> dict:
        """
        Chi-square test: Are students selected with equal frequency?
        
        H0: All students are selected with equal probability (uniform distribution)
        H1: Students are NOT selected with equal probability
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*70)
        print("1. CHI-SQUARE TEST FOR UNIFORM DISTRIBUTION")
        print("="*70)
        
        # Expected frequency under uniform distribution
        expected_per_student = self.total_selections / self.total_students
        expected_counts = np.full(self.total_students, expected_per_student)
        
        # Observed frequencies
        observed_counts = self.selection_counts
        
        # Chi-square test
        chi2_stat, p_value = chisquare(observed_counts, expected_counts)
        
        # Degrees of freedom
        df = self.total_students - 1
        
        # Critical value at alpha=0.05
        critical_value = stats.chi2.ppf(0.95, df)
        
        results = {
            'test_name': 'Chi-Square Test for Uniform Distribution',
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'critical_value': critical_value,
            'reject_null': p_value < 0.05,
            'observed_min': observed_counts.min(),
            'observed_max': observed_counts.max(),
            'observed_mean': observed_counts.mean(),
            'observed_std': observed_counts.std(),
            'expected_per_student': expected_per_student
        }
        
        # Print results
        print(f"\nNull Hypothesis (H0): All students selected with equal probability")
        print(f"Alternative Hypothesis (H1): Students NOT selected equally")
        print(f"\nObserved Selection Counts:")
        print(f"  Min: {results['observed_min']:.0f} selections")
        print(f"  Max: {results['observed_max']:.0f} selections")
        print(f"  Mean: {results['observed_mean']:.2f} selections")
        print(f"  Std: {results['observed_std']:.2f}")
        print(f"\nExpected (under uniform): {expected_per_student:.2f} per student")
        print(f"\nTest Statistics:")
        print(f"  Chi-square statistic: {chi2_stat:.4f}")
        print(f"  Degrees of freedom: {df}")
        print(f"  Critical value (α=0.05): {critical_value:.4f}")
        print(f"  P-value: {p_value:.6f}")
        
        if results['reject_null']:
            print(f"\n✗ REJECT H0 (p < 0.05)")
            print(f"  → Students are NOT selected uniformly at random")
        else:
            print(f"\n✓ FAIL TO REJECT H0 (p ≥ 0.05)")
            print(f"  → Cannot conclude students are selected non-uniformly")
        
        return results
    
    def runs_test_randomness(self) -> dict:
        """
        Runs test: Is the sequence of selections random over time?
        
        Tests if the sequence of which students are selected follows a random pattern.
        We'll test multiple aspects: runs of individual students and runs of selection patterns.
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*70)
        print("2. RUNS TEST FOR SEQUENTIAL RANDOMNESS")
        print("="*70)
        
        results = {}
        
        # Test 1: Runs test on most frequent student
        most_frequent_student = self.selection_counts.argmax()
        student_sequence = self.binary_vectors[:, most_frequent_student]
        
        # Count runs (consecutive 0s or 1s)
        runs = []
        current_run = 1
        for i in range(1, len(student_sequence)):
            if student_sequence[i] == student_sequence[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        n_runs = len(runs)
        n1 = int(student_sequence.sum())  # Number of 1s
        n0 = len(student_sequence) - n1    # Number of 0s
        
        # Expected number of runs under randomness
        if n1 > 0 and n0 > 0:
            expected_runs = (2 * n1 * n0) / (n1 + n0) + 1
            variance_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0)**2 * (n1 + n0 - 1))
            std_runs = np.sqrt(variance_runs)
            
            # Z-statistic
            z_stat = (n_runs - expected_runs) / std_runs
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = np.nan
            p_value = np.nan
            expected_runs = np.nan
        
        results['runs_test'] = {
            'test_student': most_frequent_student + 1,
            'n_runs': n_runs,
            'expected_runs': expected_runs,
            'z_statistic': z_stat,
            'p_value': p_value,
            'reject_null': p_value < 0.05 if not np.isnan(p_value) else False,
            'n_selected': n1,
            'n_not_selected': n0
        }
        
        print(f"\nTesting most frequent student (Student {most_frequent_student + 1}):")
        print(f"  Selected: {n1} times")
        print(f"  Not selected: {n0} times")
        print(f"  Observed runs: {n_runs}")
        print(f"  Expected runs (random): {expected_runs:.2f}")
        print(f"  Z-statistic: {z_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        
        if results['runs_test']['reject_null']:
            print(f"\n✗ REJECT H0 (p < 0.05)")
            print(f"  → Sequence is NOT random (shows patterns)")
        else:
            print(f"\n✓ FAIL TO REJECT H0 (p ≥ 0.05)")
            print(f"  → Sequence appears random")
        
        # Test 2: Autocorrelation test
        print(f"\n--- Autocorrelation Analysis ---")
        
        # Calculate autocorrelation for different lags
        autocorr_results = []
        for lag in [1, 7, 30]:
            if lag < len(student_sequence):
                autocorr = np.corrcoef(student_sequence[:-lag], student_sequence[lag:])[0, 1]
                autocorr_results.append({'lag': lag, 'autocorr': autocorr})
                print(f"  Lag {lag:2d}: autocorr = {autocorr:.4f}")
        
        results['autocorrelation'] = autocorr_results
        
        return results
    
    def kolmogorov_smirnov_test(self) -> dict:
        """
        Kolmogorov-Smirnov test: Compare observed distribution to uniform.
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*70)
        print("3. KOLMOGOROV-SMIRNOV TEST vs UNIFORM DISTRIBUTION")
        print("="*70)
        
        # Normalize selection counts to probabilities
        observed_probs = self.selection_counts / self.total_selections
        
        # Sort for cumulative distribution
        observed_sorted = np.sort(observed_probs)
        uniform_probs = np.ones(self.total_students) / self.total_students
        uniform_sorted = np.sort(uniform_probs)
        
        # KS test
        ks_stat, p_value = stats.ks_2samp(observed_sorted, uniform_sorted)
        
        results = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'reject_null': p_value < 0.05
        }
        
        print(f"\nNull Hypothesis (H0): Observed distribution = Uniform distribution")
        print(f"\nTest Statistics:")
        print(f"  KS statistic: {ks_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        
        if results['reject_null']:
            print(f"\n✗ REJECT H0 (p < 0.05)")
            print(f"  → Distribution is NOT uniform")
        else:
            print(f"\n✓ FAIL TO REJECT H0 (p ≥ 0.05)")
            print(f"  → Distribution is consistent with uniform")
        
        return results
    
    def inter_selection_interval_test(self) -> dict:
        """
        Test if time intervals between selections are random.
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*70)
        print("4. INTER-SELECTION INTERVAL TEST")
        print("="*70)
        
        # For each student, calculate intervals between selections
        all_intervals = []
        
        for student_id in range(self.total_students):
            selection_days = self.days[self.binary_vectors[:, student_id] == 1]
            if len(selection_days) > 1:
                intervals = np.diff(selection_days)
                all_intervals.extend(intervals)
        
        all_intervals = np.array(all_intervals)
        
        # Test if intervals follow exponential distribution (expected for random process)
        # Fit exponential distribution
        loc, scale = stats.expon.fit(all_intervals)
        
        # KS test against exponential
        ks_stat, p_value = stats.kstest(all_intervals, 'expon', args=(loc, scale))
        
        results = {
            'n_intervals': len(all_intervals),
            'mean_interval': all_intervals.mean(),
            'std_interval': all_intervals.std(),
            'median_interval': np.median(all_intervals),
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'reject_null': p_value < 0.05,
            'fitted_scale': scale
        }
        
        print(f"\nAnalyzing {len(all_intervals)} inter-selection intervals:")
        print(f"  Mean interval: {results['mean_interval']:.2f} days")
        print(f"  Median interval: {results['median_interval']:.2f} days")
        print(f"  Std deviation: {results['std_interval']:.2f} days")
        print(f"\nKS Test vs Exponential Distribution:")
        print(f"  KS statistic: {ks_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        
        if results['reject_null']:
            print(f"\n✗ REJECT H0 (p < 0.05)")
            print(f"  → Intervals are NOT consistent with random selection")
        else:
            print(f"\n✓ FAIL TO REJECT H0 (p ≥ 0.05)")
            print(f"  → Intervals consistent with random process")
        
        return results
    
    def students_per_day_variance_test(self) -> dict:
        """
        Test if number of students selected per day is constant (6).
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*70)
        print("5. STUDENTS PER DAY CONSISTENCY TEST")
        print("="*70)
        
        students_per_day = self.binary_vectors.sum(axis=1)
        
        unique_counts = np.unique(students_per_day)
        
        results = {
            'mean': students_per_day.mean(),
            'std': students_per_day.std(),
            'min': students_per_day.min(),
            'max': students_per_day.max(),
            'unique_values': unique_counts.tolist(),
            'all_equal_6': np.all(students_per_day == 6)
        }
        
        print(f"\nStudents selected per day:")
        print(f"  Mean: {results['mean']:.4f}")
        print(f"  Std: {results['std']:.4f}")
        print(f"  Min: {results['min']}")
        print(f"  Max: {results['max']}")
        print(f"  Unique values: {results['unique_values']}")
        
        if results['all_equal_6']:
            print(f"\n✓ Exactly 6 students selected every day (perfectly consistent)")
        else:
            print(f"\n✗ Number of students varies (not consistent)")
        
        return results
    
    def comprehensive_analysis(self) -> dict:
        """Run all tests and provide comprehensive summary."""
        print("\n" + "="*70)
        print("COMPREHENSIVE RANDOMNESS ANALYSIS")
        print("="*70)
        print(f"\nDataset: {self.n_samples} days, {self.total_students} students")
        print(f"Total selections: {self.total_selections}")
        print(f"Expected per student (uniform): {self.total_selections / self.total_students:.2f}")
        
        all_results = {}
        
        # Run all tests
        all_results['chi_square'] = self.chi_square_test_uniform()
        all_results['runs_test'] = self.runs_test_randomness()
        all_results['ks_test'] = self.kolmogorov_smirnov_test()
        all_results['interval_test'] = self.inter_selection_interval_test()
        all_results['consistency_test'] = self.students_per_day_variance_test()
        
        # Final conclusion
        print("\n" + "="*70)
        print("FINAL CONCLUSION")
        print("="*70)
        
        tests_reject = []
        tests_fail_to_reject = []
        
        if all_results['chi_square']['reject_null']:
            tests_reject.append("Chi-Square (frequency distribution)")
        else:
            tests_fail_to_reject.append("Chi-Square (frequency distribution)")
        
        if all_results['runs_test']['runs_test']['reject_null']:
            tests_reject.append("Runs Test (sequential patterns)")
        else:
            tests_fail_to_reject.append("Runs Test (sequential patterns)")
        
        if all_results['ks_test']['reject_null']:
            tests_reject.append("KS Test (distribution shape)")
        else:
            tests_fail_to_reject.append("KS Test (distribution shape)")
        
        if all_results['interval_test']['reject_null']:
            tests_reject.append("Interval Test (timing)")
        else:
            tests_fail_to_reject.append("Interval Test (timing)")
        
        print(f"\n📊 Test Summary:")
        print(f"  Tests rejecting randomness: {len(tests_reject)} / 4")
        print(f"  Tests supporting randomness: {len(tests_fail_to_reject)} / 4")
        
        if tests_reject:
            print(f"\n✗ Tests suggesting NON-RANDOM selection:")
            for test in tests_reject:
                print(f"    - {test}")
        
        if tests_fail_to_reject:
            print(f"\n✓ Tests consistent with RANDOM selection:")
            for test in tests_fail_to_reject:
                print(f"    - {test}")
        
        print(f"\n🎯 OVERALL CONCLUSION:")
        if len(tests_reject) >= 3:
            print(f"  ❌ STRONG EVIDENCE AGAINST RANDOMNESS")
            print(f"  → Selection process likely follows patterns or constraints")
        elif len(tests_reject) >= 2:
            print(f"  ⚠️  MODERATE EVIDENCE AGAINST RANDOMNESS")
            print(f"  → Selection may have some non-random elements")
        elif len(tests_reject) == 1:
            print(f"  ⚠️  WEAK EVIDENCE AGAINST RANDOMNESS")
            print(f"  → Mostly random with minor deviations")
        else:
            print(f"  ✅ NO EVIDENCE AGAINST RANDOMNESS")
            print(f"  → Selection appears to be random")
        
        # Additional insights
        print(f"\n💡 Key Insights:")
        chi2_pval = all_results['chi_square']['p_value']
        print(f"  • Frequency uniformity (Chi-Square p-value): {chi2_pval:.6f}")
        
        if chi2_pval < 0.05:
            print(f"    → Some students selected significantly more/less than others")
        else:
            print(f"    → All students selected with similar frequency")
        
        if all_results['consistency_test']['all_equal_6']:
            print(f"  • Exactly 6 students selected every day (high consistency)")
        
        return all_results


def main():
    """Main execution function."""
    print("="*70)
    print("STATISTICAL TESTS FOR RANDOMNESS IN STUDENT SELECTION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    # Combine for full dataset analysis
    binary_vectors = np.vstack([X_train, X_test])
    days = np.hstack([y_train, y_test])
    
    print(f"Total samples: {len(binary_vectors)}")
    print(f"Days range: {days.min()} to {days.max()}")
    
    # Initialize test suite
    test_suite = RandomnessTestSuite(binary_vectors, days, total_students=55)
    
    # Run comprehensive analysis
    results = test_suite.comprehensive_analysis()
    
    # Save results
    print("\n" + "="*70)
    print("Saving results...")
    
    summary = {
        'chi_square_pvalue': results['chi_square']['p_value'],
        'chi_square_reject': results['chi_square']['reject_null'],
        'runs_test_pvalue': results['runs_test']['runs_test']['p_value'],
        'runs_test_reject': results['runs_test']['runs_test']['reject_null'],
        'ks_test_pvalue': results['ks_test']['p_value'],
        'ks_test_reject': results['ks_test']['reject_null'],
        'interval_test_pvalue': results['interval_test']['p_value'],
        'interval_test_reject': results['interval_test']['reject_null']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('randomness_test_results.csv', index=False)
    print("✓ Saved: randomness_test_results.csv")
    
    return results


if __name__ == "__main__":
    results = main()
