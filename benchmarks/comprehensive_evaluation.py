"""
Comprehensive SPaR-K Evaluation Framework

This module runs all benchmarks and provides systematic comparison between
SPaR-K and standard Transformer baselines across multiple reasoning tasks.

Includes statistical significance testing and detailed performance analysis.
"""

import torch
import numpy as np
import json
import time
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from tqdm import tqdm

from multi_hop_reasoning import MultiHopBenchmark
from spd_router_evaluation import SPDEvaluationBenchmark
from verifier_head_evaluation import VerifierHeadBenchmark


class ComprehensiveBenchmark:
    """Master benchmark runner for complete SPaR-K evaluation"""
    
    def __init__(self, device='cpu', num_trials=3):
        self.device = device
        self.num_trials = num_trials  # Multiple runs for statistical significance
        self.results = {}
        
    def run_all_benchmarks(self, quick_mode=False):
        """Run all benchmark suites with multiple trials"""
        print("🔬 COMPREHENSIVE SPaR-K EVALUATION")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Trials per benchmark: {self.num_trials}")
        print(f"Mode: {'Quick' if quick_mode else 'Full'}")
        print()
        
        all_results = {}
        
        # 1. Multi-hop Reasoning Evaluation
        print("\n📊 1. MULTI-HOP REASONING EVALUATION")
        print("-" * 40)
        
        multihop_results = []
        for trial in range(self.num_trials):
            print(f"Trial {trial + 1}/{self.num_trials}")
            benchmark = MultiHopBenchmark(device=self.device)
            
            try:
                result = benchmark.run_comprehensive_evaluation()
                multihop_results.append(result)
                print(f"✅ Trial {trial + 1} completed")
            except Exception as e:
                print(f"❌ Trial {trial + 1} failed: {e}")
                multihop_results.append(None)
        
        all_results['multi_hop'] = multihop_results
        
        # 2. SPD Router Evaluation  
        print("\n📈 2. SPD ROUTER EVALUATION")
        print("-" * 40)
        
        spd_results = []
        for trial in range(self.num_trials):
            print(f"Trial {trial + 1}/{self.num_trials}")
            benchmark = SPDEvaluationBenchmark(device=self.device)
            
            try:
                result = benchmark.run_comprehensive_evaluation()
                spd_results.append(result)
                print(f"✅ Trial {trial + 1} completed")
            except Exception as e:
                print(f"❌ Trial {trial + 1} failed: {e}")
                spd_results.append(None)
        
        all_results['spd_router'] = spd_results
        
        # 3. Verifier Head Evaluation
        print("\n🔍 3. VERIFIER HEAD EVALUATION")
        print("-" * 40)
        
        verifier_results = []
        for trial in range(self.num_trials):
            print(f"Trial {trial + 1}/{self.num_trials}")
            benchmark = VerifierHeadBenchmark(device=self.device)
            
            try:
                result = benchmark.run_comprehensive_evaluation()
                verifier_results.append(result)
                print(f"✅ Trial {trial + 1} completed")
            except Exception as e:
                print(f"❌ Trial {trial + 1} failed: {e}")
                verifier_results.append(None)
        
        all_results['verifier_head'] = verifier_results
        
        # Aggregate and analyze results
        self.results = all_results
        analysis = self.analyze_results()
        
        return all_results, analysis
    
    def analyze_results(self):
        """Analyze results across all trials for statistical significance"""
        print("\n📊 STATISTICAL ANALYSIS")
        print("="*60)
        
        analysis = {}
        
        # Analyze multi-hop results
        if 'multi_hop' in self.results:
            multihop_analysis = self._analyze_multihop_results()
            analysis['multi_hop'] = multihop_analysis
        
        # Analyze SPD results
        if 'spd_router' in self.results:
            spd_analysis = self._analyze_spd_results()
            analysis['spd_router'] = spd_analysis
        
        # Analyze Verifier results
        if 'verifier_head' in self.results:
            verifier_analysis = self._analyze_verifier_results()
            analysis['verifier_head'] = verifier_analysis
        
        return analysis
    
    def _analyze_multihop_results(self):
        """Analyze multi-hop reasoning results"""
        valid_results = [r for r in self.results['multi_hop'] if r is not None]
        
        if not valid_results:
            return {"error": "No valid multi-hop results"}
        
        print("\nMulti-hop Reasoning Analysis:")
        
        # Extract graph traversal accuracies
        spark_accuracies = []
        baseline_accuracies = []
        
        for result in valid_results:
            if 'spark' in result and 'baseline' in result:
                # Calculate average accuracy across path lengths
                spark_avg = np.mean(list(result['spark']['graph_traversal'].values()))
                baseline_avg = np.mean(list(result['baseline']['graph_traversal'].values()))
                
                spark_accuracies.append(spark_avg)
                baseline_accuracies.append(baseline_avg)
        
        if spark_accuracies and baseline_accuracies:
            # Statistical test
            t_stat, p_value = stats.ttest_rel(spark_accuracies, baseline_accuracies)
            
            spark_mean = np.mean(spark_accuracies)
            baseline_mean = np.mean(baseline_accuracies)
            improvement = spark_mean - baseline_mean
            
            print(f"  SPaR-K accuracy: {spark_mean:.3f} ± {np.std(spark_accuracies):.3f}")
            print(f"  Baseline accuracy: {baseline_mean:.3f} ± {np.std(baseline_accuracies):.3f}")
            print(f"  Improvement: {improvement:.3f} (p-value: {p_value:.4f})")
            
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"  Statistical significance: {significance}")
            
            return {
                'spark_mean': spark_mean,
                'baseline_mean': baseline_mean,
                'improvement': improvement,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'num_trials': len(spark_accuracies)
            }
        
        return {"error": "Insufficient data for analysis"}
    
    def _analyze_spd_results(self):
        """Analyze SPD router results"""
        valid_results = [r for r in self.results['spd_router'] if r is not None]
        
        if not valid_results:
            return {"error": "No valid SPD results"}
        
        print("\nSPD Router Analysis:")
        
        # Aggregate across datasets and methods
        all_improvements = []
        all_correlations = []
        
        for result in valid_results:
            for dataset_name, dataset_results in result.items():
                if 'standard_spd' in dataset_results:
                    spd_result = dataset_results['standard_spd']
                    baseline_result = dataset_results.get('baseline_filter', {})
                    
                    # SNR improvement
                    if 'snr_improvement_db' in spd_result:
                        all_improvements.append(spd_result['snr_improvement_db'])
                    
                    # Structure correlation
                    if 'structure_correlation' in spd_result:
                        all_correlations.append(spd_result['structure_correlation'])
        
        if all_improvements:
            mean_snr_improvement = np.mean(all_improvements)
            std_snr_improvement = np.std(all_improvements)
            
            print(f"  SNR improvement: {mean_snr_improvement:.2f} ± {std_snr_improvement:.2f} dB")
            
            # Test if improvement is significant
            t_stat, p_value = stats.ttest_1samp(all_improvements, 0)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"  Improvement significance: {significance} (p={p_value:.4f})")
            
            return {
                'mean_snr_improvement': mean_snr_improvement,
                'std_snr_improvement': std_snr_improvement,
                'mean_structure_correlation': np.mean(all_correlations) if all_correlations else 0,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return {"error": "Insufficient data for analysis"}
    
    def _analyze_verifier_results(self):
        """Analyze verifier head results"""
        valid_results = [r for r in self.results['verifier_head'] if r is not None]
        
        if not valid_results:
            return {"error": "No valid verifier results"}
        
        print("\nVerifier Head Analysis:")
        
        # Extract generalization gaps
        spark_gaps = []
        baseline_gaps = []
        
        for result in valid_results:
            for dataset_name, dataset_results in result.items():
                if 'spark' in dataset_results and 'baseline' in dataset_results:
                    spark_gap = dataset_results['spark']['generalization_gap']
                    baseline_gap = dataset_results['baseline']['generalization_gap']
                    
                    spark_gaps.append(spark_gap)
                    baseline_gaps.append(baseline_gap)
        
        if spark_gaps and baseline_gaps:
            spark_mean_gap = np.mean(spark_gaps)
            baseline_mean_gap = np.mean(baseline_gaps)
            
            # Lower gap is better (smaller difference between train/test)
            gap_reduction = baseline_mean_gap - spark_mean_gap
            
            t_stat, p_value = stats.ttest_rel(baseline_gaps, spark_gaps)
            
            print(f"  SPaR-K generalization gap: {spark_mean_gap:.3f} ± {np.std(spark_gaps):.3f}")
            print(f"  Baseline generalization gap: {baseline_mean_gap:.3f} ± {np.std(baseline_gaps):.3f}")
            print(f"  Gap reduction: {gap_reduction:.3f} (p-value: {p_value:.4f})")
            
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"  Statistical significance: {significance}")
            
            return {
                'spark_mean_gap': spark_mean_gap,
                'baseline_mean_gap': baseline_mean_gap,
                'gap_reduction': gap_reduction,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return {"error": "Insufficient data for analysis"}
    
    def create_comprehensive_report(self, analysis):
        """Create detailed performance report"""
        print("\n📋 COMPREHENSIVE PERFORMANCE REPORT")
        print("="*60)
        
        report = {
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'num_trials': self.num_trials,
            'analysis': analysis
        }
        
        print("\n🎯 SUMMARY OF FINDINGS:")
        
        # Multi-hop reasoning
        if 'multi_hop' in analysis and 'improvement' in analysis['multi_hop']:
            multihop = analysis['multi_hop']
            print(f"\n1. Multi-hop Reasoning:")
            print(f"   SPaR-K vs Baseline: {multihop['improvement']:+.3f} accuracy improvement")
            print(f"   Statistical significance: {'Yes' if multihop.get('significant', False) else 'No'} (p={multihop.get('p_value', 'N/A'):.4f})")
            print(f"   Based on {multihop.get('num_trials', 0)} trials")
        
        # SPD Router
        if 'spd_router' in analysis and 'mean_snr_improvement' in analysis['spd_router']:
            spd = analysis['spd_router']
            print(f"\n2. Structure/Noise Separation:")
            print(f"   SNR improvement: {spd['mean_snr_improvement']:+.2f} ± {spd['std_snr_improvement']:.2f} dB")
            print(f"   Structure correlation: {spd.get('mean_structure_correlation', 0):.3f}")
            print(f"   Statistical significance: {'Yes' if spd.get('significant', False) else 'No'} (p={spd.get('p_value', 'N/A'):.4f})")
        
        # Verifier Head
        if 'verifier_head' in analysis and 'gap_reduction' in analysis['verifier_head']:
            verifier = analysis['verifier_head']
            print(f"\n3. Algorithmic Generalization:")
            print(f"   Generalization gap reduction: {verifier['gap_reduction']:+.3f}")
            print(f"   SPaR-K gap: {verifier['spark_mean_gap']:.3f}")
            print(f"   Baseline gap: {verifier['baseline_mean_gap']:.3f}")
            print(f"   Statistical significance: {'Yes' if verifier.get('significant', False) else 'No'} (p={verifier.get('p_value', 'N/A'):.4f})")
        
        # Overall assessment
        print(f"\n🎓 OVERALL ASSESSMENT:")
        
        significant_improvements = 0
        total_tests = 0
        
        for component, component_analysis in analysis.items():
            if isinstance(component_analysis, dict) and 'significant' in component_analysis:
                total_tests += 1
                if component_analysis['significant']:
                    significant_improvements += 1
        
        print(f"   Significant improvements: {significant_improvements}/{total_tests} tests")
        
        if significant_improvements == 0:
            print(f"   ⚠️  No statistically significant improvements found")
            print(f"   🔬 Architecture demonstrates feasibility but requires further optimization")
        elif significant_improvements == total_tests:
            print(f"   ✅ All components show significant improvements")
            print(f"   🚀 Architecture ready for broader evaluation")
        else:
            print(f"   ⚡ Mixed results - some components show promise")
            print(f"   🔧 Selective adoption recommended based on task requirements")
        
        return report
    
    def visualize_results(self, analysis):
        """Create comprehensive visualization of results"""
        print("\n📊 Creating performance visualizations...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Multi-hop reasoning comparison
        if 'multi_hop' in analysis and 'improvement' in analysis['multi_hop']:
            multihop = analysis['multi_hop']
            
            models = ['Baseline', 'SPaR-K']
            accuracies = [multihop['baseline_mean'], multihop['spark_mean']]
            
            bars = ax1.bar(models, accuracies, color=['red', 'blue'], alpha=0.7)
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Multi-hop Reasoning Performance')
            ax1.set_ylim(0, 1)
            
            # Add significance indicator
            if multihop.get('significant', False):
                ax1.text(0.5, max(accuracies) + 0.05, f"p={multihop.get('p_value', 0):.3f}*", 
                        ha='center', fontweight='bold')
            
            # Add values on bars
            for bar, acc in zip(bars, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'Multi-hop\nEvaluation\nFailed', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Multi-hop Reasoning (Failed)')
        
        # 2. SNR improvement
        if 'spd_router' in analysis and 'mean_snr_improvement' in analysis['spd_router']:
            spd = analysis['spd_router']
            
            improvement = spd['mean_snr_improvement']
            error = spd['std_snr_improvement']
            
            bars = ax2.bar(['SPD Router'], [improvement], yerr=[error], 
                          color='orange', alpha=0.7, capsize=5)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_ylabel('SNR Improvement (dB)')
            ax2.set_title('Signal-to-Noise Ratio Enhancement')
            
            if spd.get('significant', False):
                ax2.text(0, improvement + error + 0.5, f"p={spd.get('p_value', 0):.3f}*", 
                        ha='center', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'SPD Router\nEvaluation\nFailed', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('SPD Router (Failed)')
        
        # 3. Generalization gap comparison
        if 'verifier_head' in analysis and 'gap_reduction' in analysis['verifier_head']:
            verifier = analysis['verifier_head']
            
            models = ['Baseline', 'SPaR-K']
            gaps = [verifier['baseline_mean_gap'], verifier['spark_mean_gap']]
            
            bars = ax3.bar(models, gaps, color=['red', 'green'], alpha=0.7)
            ax3.set_ylabel('Generalization Gap')
            ax3.set_title('Algorithmic Generalization')
            
            # Lower is better for generalization gap
            if verifier.get('significant', False):
                ax3.text(0.5, max(gaps) + 0.02, f"p={verifier.get('p_value', 0):.3f}*", 
                        ha='center', fontweight='bold')
            
            for bar, gap in zip(bars, gaps):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                        f'{gap:.3f}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'Verifier Head\nEvaluation\nFailed', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Verifier Head (Failed)')
        
        # 4. Overall summary
        components = []
        improvements = []
        significances = []
        
        if 'multi_hop' in analysis and 'improvement' in analysis['multi_hop']:
            components.append('Multi-hop\nReasoning')
            improvements.append(analysis['multi_hop']['improvement'])
            significances.append(analysis['multi_hop'].get('significant', False))
        
        if 'spd_router' in analysis and 'mean_snr_improvement' in analysis['spd_router']:
            components.append('SPD Router\n(SNR dB)')
            improvements.append(analysis['spd_router']['mean_snr_improvement'])
            significances.append(analysis['spd_router'].get('significant', False))
        
        if 'verifier_head' in analysis and 'gap_reduction' in analysis['verifier_head']:
            components.append('Verifier Head\n(Gap Reduction)')
            improvements.append(analysis['verifier_head']['gap_reduction'])
            significances.append(analysis['verifier_head'].get('significant', False))
        
        if components:
            colors = ['green' if sig else 'orange' for sig in significances]
            bars = ax4.bar(components, improvements, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_ylabel('Improvement')
            ax4.set_title('Component Performance Summary')
            
            # Add significance indicators
            for i, (bar, improvement, sig) in enumerate(zip(bars, improvements, significances)):
                marker = '*' if sig else ''
                ax4.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (0.1 if improvement > 0 else -0.1), 
                        f'{improvement:.2f}{marker}', ha='center', va='bottom' if improvement > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig('comprehensive_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'comprehensive_benchmark_results.png'")
    
    def save_detailed_results(self, analysis):
        """Save detailed results and analysis"""
        detailed_report = {
            'metadata': {
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'num_trials': self.num_trials,
                'spark_version': '1.0.0'
            },
            'raw_results': self.results,
            'statistical_analysis': analysis
        }
        
        # Save to JSON
        with open('comprehensive_evaluation_results.json', 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        
        # Create summary CSV
        summary_data = []
        
        for component, component_analysis in analysis.items():
            if isinstance(component_analysis, dict) and not 'error' in component_analysis:
                row = {
                    'Component': component.replace('_', ' ').title(),
                    'Metric': self._get_primary_metric(component),
                    'Improvement': self._get_improvement_value(component_analysis),
                    'P_Value': component_analysis.get('p_value', 'N/A'),
                    'Significant': component_analysis.get('significant', False),
                    'Status': 'Significant' if component_analysis.get('significant', False) else 'Not Significant'
                }
                summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv('benchmark_summary.csv', index=False)
            print("Summary saved as 'benchmark_summary.csv'")
        
        print("Detailed results saved as 'comprehensive_evaluation_results.json'")
    
    def _get_primary_metric(self, component):
        """Get primary metric name for component"""
        metric_map = {
            'multi_hop': 'Accuracy Improvement',
            'spd_router': 'SNR Improvement (dB)',
            'verifier_head': 'Generalization Gap Reduction'
        }
        return metric_map.get(component, 'Unknown')
    
    def _get_improvement_value(self, analysis):
        """Extract primary improvement value"""
        if 'improvement' in analysis:
            return f"{analysis['improvement']:+.3f}"
        elif 'mean_snr_improvement' in analysis:
            return f"{analysis['mean_snr_improvement']:+.2f}"
        elif 'gap_reduction' in analysis:
            return f"{analysis['gap_reduction']:+.3f}"
        else:
            return "N/A"


def main():
    """Run comprehensive evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run evaluation
    benchmark = ComprehensiveBenchmark(device=device, num_trials=2)  # Reduced for testing
    
    start_time = time.time()
    raw_results, analysis = benchmark.run_all_benchmarks()
    end_time = time.time()
    
    print(f"\n⏱️  Total evaluation time: {end_time - start_time:.1f} seconds")
    
    # Create report and visualizations
    report = benchmark.create_comprehensive_report(analysis)
    benchmark.visualize_results(analysis)
    benchmark.save_detailed_results(analysis)
    
    print(f"\n🎉 COMPREHENSIVE EVALUATION COMPLETE!")
    print(f"📁 Results saved to:")
    print(f"   • comprehensive_evaluation_results.json (detailed)")
    print(f"   • benchmark_summary.csv (summary)")  
    print(f"   • comprehensive_benchmark_results.png (visualization)")


if __name__ == "__main__":
    main()