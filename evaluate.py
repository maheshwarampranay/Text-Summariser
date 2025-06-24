
import evaluate
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SummarizationEvaluator:
    """Comprehensive evaluation for text summarization models"""

    def __init__(self):
        self.rouge_metric = evaluate.load("rouge")
        self.meteor_metric = evaluate.load("meteor")
        self.bleu_metric = evaluate.load("bleu")

    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores"""
        try:
            results = self.rouge_metric.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True
            )

            return {
                'rouge1': results['rouge1'] * 100,
                'rouge2': results['rouge2'] * 100,
                'rougeL': results['rougeL'] * 100,
                'rougeLsum': results['rougeLsum'] * 100
            }
        except Exception as e:
            logger.error(f"Error computing ROUGE scores: {e}")
            return {}

    def compute_meteor_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute METEOR score"""
        try:
            result = self.meteor_metric.compute(
                predictions=predictions,
                references=references
            )
            return result['meteor'] * 100
        except Exception as e:
            logger.error(f"Error computing METEOR score: {e}")
            return 0.0

    def compute_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score"""
        try:
            # Prepare references in the format expected by BLEU
            references_formatted = [[ref.split()] for ref in references]
            predictions_formatted = [pred.split() for pred in predictions]

            result = self.bleu_metric.compute(
                predictions=predictions_formatted,
                references=references_formatted
            )
            return result['bleu'] * 100
        except Exception as e:
            logger.error(f"Error computing BLEU score: {e}")
            return 0.0

    def compute_length_stats(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute length-based statistics"""
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]

        return {
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths),
            'compression_ratio': np.mean(ref_lengths) / np.mean(pred_lengths)
        }

    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        logger.info("Computing evaluation metrics...")

        metrics = {}

        # ROUGE scores
        rouge_scores = self.compute_rouge_scores(predictions, references)
        metrics.update(rouge_scores)

        # METEOR score
        meteor_score = self.compute_meteor_score(predictions, references)
        metrics['meteor'] = meteor_score

        # BLEU score
        bleu_score = self.compute_bleu_score(predictions, references)
        metrics['bleu'] = bleu_score

        # Length statistics
        length_stats = self.compute_length_stats(predictions, references)
        metrics.update(length_stats)

        return metrics

    def print_metrics(self, metrics: Dict[str, float]):
        """Pretty print evaluation metrics"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)

        print(f"ROUGE-1:     {metrics.get('rouge1', 0):.2f}")
        print(f"ROUGE-2:     {metrics.get('rouge2', 0):.2f}")
        print(f"ROUGE-L:     {metrics.get('rougeL', 0):.2f}")
        print(f"ROUGE-Lsum:  {metrics.get('rougeLsum', 0):.2f}")
        print(f"METEOR:      {metrics.get('meteor', 0):.2f}")
        print(f"BLEU:        {metrics.get('bleu', 0):.2f}")

        print("\n" + "-"*30)
        print("LENGTH STATISTICS")
        print("-"*30)
        print(f"Avg Prediction Length: {metrics.get('avg_pred_length', 0):.1f} words")
        print(f"Avg Reference Length:  {metrics.get('avg_ref_length', 0):.1f} words")
        print(f"Length Ratio:          {metrics.get('length_ratio', 0):.2f}")
        print(f"Compression Ratio:     {metrics.get('compression_ratio', 0):.2f}")
        print("="*50)

class ModelComparator:
    """Compare multiple summarization models"""

    def __init__(self):
        self.evaluator = SummarizationEvaluator()
        self.results = {}

    def add_model_results(self, model_name: str, predictions: List[str], references: List[str]):
        """Add results for a model"""
        metrics = self.evaluator.compute_all_metrics(predictions, references)
        self.results[model_name] = metrics
        logger.info(f"Added results for {model_name}")

    def compare_models(self):
        """Compare all added models"""
        if not self.results:
            logger.warning("No model results to compare")
            return

        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        # Print header
        models = list(self.results.keys())
        print(f"{'Metric':<15}", end="")
        for model in models:
            print(f"{model:<15}", end="")
        print()
        print("-" * (15 + 15 * len(models)))

        # Print metrics
        key_metrics = ['rouge1', 'rouge2', 'rougeL', 'meteor', 'bleu']
        for metric in key_metrics:
            print(f"{metric.upper():<15}", end="")
            for model in models:
                value = self.results[model].get(metric, 0)
                print(f"{value:<15.2f}", end="")
            print()

        print("="*80)

        # Find best model for each metric
        print("\nBEST MODELS PER METRIC:")
        for metric in key_metrics:
            best_model = max(models, key=lambda m: self.results[m].get(metric, 0))
            best_score = self.results[best_model].get(metric, 0)
            print(f"{metric.upper()}: {best_model} ({best_score:.2f})")

def evaluate_model_predictions(predictions_file: str, references_file: str):
    """Evaluate predictions from file"""
    try:
        with open(predictions_file, 'r') as f:
            predictions = [line.strip() for line in f.readlines()]

        with open(references_file, 'r') as f:
            references = [line.strip() for line in f.readlines()]

        if len(predictions) != len(references):
            logger.error("Predictions and references must have the same length")
            return

        evaluator = SummarizationEvaluator()
        metrics = evaluator.compute_all_metrics(predictions, references)
        evaluator.print_metrics(metrics)

        return metrics

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Error evaluating predictions: {e}")
