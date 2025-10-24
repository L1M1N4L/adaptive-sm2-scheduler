"""
Comprehensive Evaluation Script

This script integrates our SM-2 implementation with the SSP-MMC-Plus evaluation framework
to run comprehensive spaced repetition algorithm evaluation following the master.md research plan.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import time
import argparse

# Add our src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.schedulers import SM2Scheduler, RatingConverter
from scripts.ssp_mmc_integration import SM2SSPMMCAdapter


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator that integrates our SM-2 with SSP-MMC-Plus framework.
    
    This class provides a complete evaluation pipeline following the master.md research phases.
    """
    
    def __init__(self, ssp_mmc_path="./evaluation/SSP-MMC-Plus"):
        """
        Initialize the comprehensive evaluator.
        
        Args:
            ssp_mmc_path: Path to SSP-MMC-Plus framework
        """
        self.ssp_mmc_path = ssp_mmc_path
        self.adapter = SM2SSPMMCAdapter()
        self.results = {}
        
        # Add SSP-MMC-Plus to Python path
        if os.path.exists(ssp_mmc_path):
            sys.path.append(ssp_mmc_path)
            sys.path.append(os.path.join(ssp_mmc_path, 'model'))
    
    def load_sample_data(self):
        """
        Load sample data for testing the evaluation pipeline.
        
        Returns:
            DataFrame with sample SSP-MMC-Plus format data
        """
        print("Creating sample data for evaluation testing...")
        
        # Create sample data in SSP-MMC-Plus format
        sample_data = pd.DataFrame({
            'u': ['user1', 'user1', 'user2', 'user2'],
            'w': ['word1', 'word2', 'word1', 'word3'],
            'i': [1, 1, 2, 1],
            'd': [5, 3, 5, 4],
            't_history': ['[]', '[]', '[1, 6]', '[]'],
            'r_history': ['[]', '[]', '[1, 1]', '[]'],
            'p_history': ['[]', '[]', '[0.9, 0.8]', '[]'],  # Add missing p_history
            'delta_t': [1, 1, 7, 1],
            'r': [1, 1, 1, 0],
            'p': [0.9, 0.8, 0.7, 0.3],
            'h': [10.0, 8.0, 15.0, 2.0],
            'total_cnt': [100, 150, 200, 50]
        })
        
        return sample_data
    
    def run_phase_1_evaluation(self):
        """
        Run Phase 1 evaluation: SM-2 baseline implementation.
        
        This evaluates our SM-2 implementation using the SSP-MMC-Plus framework.
        """
        print("=" * 60)
        print("PHASE 1 EVALUATION: SM-2 Baseline Implementation")
        print("=" * 60)
        
        # Load sample data
        sample_data = self.load_sample_data()
        
        # Run evaluation
        results = self.adapter.evaluate_sm2_on_ssp_mmc(sample_data, repeat=1, fold=1)
        
        # Calculate metrics
        mae_p = np.mean(np.abs(results['p'] - results['pp']))
        mae_h = np.mean(np.abs(results['h'] - results['hh']))
        
        print(f"\nPhase 1 Results:")
        print(f"  MAE (Recall Probability): {mae_p:.4f}")
        print(f"  MAE (Half-life): {mae_h:.4f}")
        print(f"  Samples Processed: {len(results)}")
        
        self.results['phase_1'] = {
            'mae_p': mae_p,
            'mae_h': mae_h,
            'samples': len(results)
        }
        
        return results
    
    def run_ssp_mmc_metrics(self, results):
        """
        Calculate SSP-MMC-Plus evaluation metrics.
        
        Args:
            results: Evaluation results DataFrame
            
        Returns:
            Dictionary of calculated metrics
        """
        print("\nCalculating SSP-MMC-Plus Metrics...")
        
        # Check if we have valid results
        if len(results) == 0:
            print("  No valid results to calculate metrics")
            return {
                'mae_p': np.nan, 'mse_p': np.nan, 'mae_h': np.nan, 'mse_h': np.nan,
                'corr_p': np.nan, 'corr_h': np.nan, 'r2_p': np.nan, 'r2_h': np.nan
            }
        
        # Calculate various metrics
        metrics = {}
        
        # Basic prediction accuracy
        metrics['mae_p'] = np.mean(np.abs(results['p'] - results['pp']))
        metrics['mse_p'] = np.mean((results['p'] - results['pp']) ** 2)
        metrics['mae_h'] = np.mean(np.abs(results['h'] - results['hh']))
        metrics['mse_h'] = np.mean((results['h'] - results['hh']) ** 2)
        
        # Correlation metrics (handle edge cases)
        try:
            if len(results) > 1 and np.std(results['p']) > 0 and np.std(results['pp']) > 0:
                metrics['corr_p'] = np.corrcoef(results['p'], results['pp'])[0, 1]
            else:
                metrics['corr_p'] = np.nan
        except:
            metrics['corr_p'] = np.nan
            
        try:
            if len(results) > 1 and np.std(results['h']) > 0 and np.std(results['hh']) > 0:
                metrics['corr_h'] = np.corrcoef(results['h'], results['hh'])[0, 1]
            else:
                metrics['corr_h'] = np.nan
        except:
            metrics['corr_h'] = np.nan
        
        # R-squared
        ss_res_p = np.sum((results['p'] - results['pp']) ** 2)
        ss_tot_p = np.sum((results['p'] - np.mean(results['p'])) ** 2)
        metrics['r2_p'] = 1 - (ss_res_p / ss_tot_p) if ss_tot_p > 0 else 0
        
        ss_res_h = np.sum((results['h'] - results['hh']) ** 2)
        ss_tot_h = np.sum((results['h'] - np.mean(results['h'])) ** 2)
        metrics['r2_h'] = 1 - (ss_res_h / ss_tot_h) if ss_tot_h > 0 else 0
        
        print(f"  MAE (p): {metrics['mae_p']:.4f}")
        print(f"  MSE (p): {metrics['mse_p']:.4f}")
        print(f"  R² (p): {metrics['r2_p']:.4f}")
        print(f"  MAE (h): {metrics['mae_h']:.4f}")
        print(f"  MSE (h): {metrics['mse_h']:.4f}")
        print(f"  R² (h): {metrics['r2_h']:.4f}")
        
        return metrics
    
    def generate_evaluation_report(self):
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            String containing the evaluation report
        """
        report = []
        report.append("COMPREHENSIVE EVALUATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if 'phase_1' in self.results:
            report.append("PHASE 1: SM-2 Baseline Implementation")
            report.append("-" * 40)
            phase1 = self.results['phase_1']
            report.append(f"  MAE (Recall Probability): {phase1['mae_p']:.4f}")
            report.append(f"  MAE (Half-life): {phase1['mae_h']:.4f}")
            report.append(f"  Samples Processed: {phase1['samples']}")
            report.append("")
        
        report.append("EVALUATION STATUS")
        report.append("-" * 40)
        report.append("  Phase 1 (SM-2 Baseline): COMPLETE")
        report.append("  Phase 2 (ML Models): PENDING")
        report.append("  Phase 3 (Hybrid Scheduler): PENDING")
        report.append("  Phase 4 (Comprehensive Evaluation): IN PROGRESS")
        report.append("  Phase 5 (IoT Integration): PENDING")
        report.append("")
        
        report.append("NEXT STEPS")
        report.append("-" * 40)
        report.append("1. Download FSRS-Anki-20k dataset")
        report.append("2. Implement ML-based schedulers")
        report.append("3. Create hybrid SM-2+AI scheduler")
        report.append("4. Run full SSP-MMC-Plus evaluation")
        report.append("5. Deploy on IoT tactile flashcard hardware")
        
        return "\n".join(report)
    
    def run_complete_evaluation(self):
        """
        Run the complete evaluation pipeline.
        
        This is the main method that orchestrates the entire evaluation process.
        """
        print("Starting Comprehensive Evaluation Pipeline")
        print("Following master.md research phases...")
        print()
        
        # Phase 1: SM-2 Baseline
        phase1_results = self.run_phase_1_evaluation()
        
        # Calculate SSP-MMC-Plus metrics
        metrics = self.run_ssp_mmc_metrics(phase1_results)
        
        # Generate report
        report = self.generate_evaluation_report()
        print("\n" + report)
        
        # Save results
        self.save_results(phase1_results, metrics)
        
        return phase1_results, metrics
    
    def save_results(self, results, metrics):
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results DataFrame
            metrics: Calculated metrics dictionary
        """
        # Create results directory
        results_dir = Path("./evaluation/results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        timestamp = int(time.time())
        results_file = results_dir / f"sm2_evaluation_{timestamp}.csv"
        results.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Save metrics
        metrics_file = results_dir / f"sm2_metrics_{timestamp}.json"
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")
        
        # Save report
        report_file = results_dir / f"evaluation_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(self.generate_evaluation_report())
        print(f"Report saved to: {report_file}")


def main():
    """Main function to run the comprehensive evaluation."""
    parser = argparse.ArgumentParser(description='Run comprehensive SM-2 evaluation')
    parser.add_argument('--ssp-mmc-path', default='./SSP-MMC-Plus',
                       help='Path to SSP-MMC-Plus framework')
    parser.add_argument('--output-dir', default='./evaluation/results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(args.ssp_mmc_path)
    
    # Run complete evaluation
    results, metrics = evaluator.run_complete_evaluation()
    
    print("\nEvaluation completed successfully!")
    print("Check the evaluation/results directory for detailed outputs.")


if __name__ == "__main__":
    main()
