
"""
Comprehensive Demo Script for Text Summarization
Shows the complete workflow from training to inference
"""

import os
import logging
from text_summarizer_main import TextSummarizer, ModelConfig
from data_loader import DatasetLoader
from evaluate import SummarizationEvaluator, ModelComparator
from inference import TextSummarizerInference

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_data_loading():
    """Demo data loading capabilities"""
    print("\n" + "="*60)
    print("DEMO: Data Loading and Preprocessing")
    print("="*60)

    # Show available datasets
    datasets = ['cnn_dailymail', 'xsum', 'billsum']

    for dataset_name in datasets:
        info = DatasetLoader.get_dataset_info(dataset_name)
        if info:
            print(f"\n{dataset_name.upper()}:")
            for key, value in info.items():
                print(f"  {key}: {value}")

def demo_model_comparison():
    """Demo model comparison with synthetic results"""
    print("\n" + "="*60)
    print("DEMO: Model Comparison")
    print("="*60)

    # Create synthetic evaluation results for demo
    comparator = ModelComparator()

    # Add synthetic results (based on research findings)
    models_results = {
        'T5-Small': {
            'rouge1': 35.4, 'rouge2': 14.0, 'rougeL': 28.5, 
            'meteor': 35.0, 'bleu': 12.8
        },
        'BART-Base': {
            'rouge1': 30.8, 'rouge2': 15.7, 'rougeL': 26.2, 
            'meteor': 28.0, 'bleu': 14.2
        },
        'PEGASUS-XSum': {
            'rouge1': 24.5, 'rouge2': 12.0, 'rougeL': 21.8, 
            'meteor': 25.0, 'bleu': 10.5
        }
    }

    for model_name, results in models_results.items():
        comparator.results[model_name] = results

    comparator.compare_models()

def demo_inference():
    """Demo inference pipeline"""
    print("\n" + "="*60)
    print("DEMO: Text Summarization Inference")
    print("="*60)

    # Sample texts for summarization
    sample_texts = [
        """
        Artificial Intelligence has become one of the most transformative technologies of the 21st century. 
        From machine learning algorithms that power recommendation systems to deep learning models that can 
        generate human-like text, AI is reshaping industries and changing how we interact with technology. 
        The field has seen rapid advancement in recent years, particularly with the development of large 
        language models like GPT-3 and BERT. These models have demonstrated remarkable capabilities in 
        understanding and generating natural language, leading to applications in chatbots, content creation, 
        and automated customer service. However, the development of AI also raises important ethical 
        considerations, including concerns about bias, privacy, and the potential impact on employment. 
        As AI continues to evolve, it's crucial that we develop frameworks for responsible AI deployment 
        and ensure that these powerful technologies benefit society as a whole.
        """,
        """
        Climate change represents one of the most pressing challenges facing humanity today. Rising global 
        temperatures, caused primarily by greenhouse gas emissions from human activities, are leading to 
        significant environmental changes worldwide. These include melting polar ice caps, rising sea levels, 
        more frequent extreme weather events, and shifts in precipitation patterns. The impacts extend beyond 
        environmental concerns to affect human health, food security, water resources, and economic stability. 
        Scientists have reached a strong consensus that immediate action is needed to reduce greenhouse gas 
        emissions and transition to renewable energy sources. Governments, businesses, and individuals all 
        have important roles to play in addressing this challenge through policy changes, technological 
        innovation, and behavioral modifications. The next decade is considered critical for implementing 
        effective climate action to limit global warming and mitigate its most severe consequences.
        """
    ]

    try:
        # Initialize summarizer with a pre-trained model
        summarizer = TextSummarizerInference(model_name="t5-small")

        print("Model Information:")
        info = summarizer.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("\nGenerating Summaries...")
        print("-" * 40)

        for i, text in enumerate(sample_texts, 1):
            print(f"\nText {i}:")
            print(f"Original ({len(text.split())} words):")
            print(text.strip()[:200] + "..." if len(text) > 200 else text.strip())

            summary = summarizer.summarize(
                text.strip(),
                max_length=80,
                min_length=20,
                num_beams=4
            )

            print(f"\nSummary ({len(summary.split())} words):")
            print(summary)
            print("-" * 40)

    except Exception as e:
        logger.error(f"Error in inference demo: {e}")
        print("Note: Inference demo requires model files. Run training first or use pre-trained models.")

def demo_training_workflow():
    """Demo the training workflow (simplified)"""
    print("\n" + "="*60)
    print("DEMO: Training Workflow")
    print("="*60)

    print("Training Configuration:")
    config = ModelConfig(
        model_name="t5-small",
        dataset_name="cnn_dailymail",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        output_dir="./demo_model"
    )

    # Print configuration
    for field, value in config.__dict__.items():
        print(f"  {field}: {value}")

    print("\nTraining Steps:")
    print("1. Load and preprocess dataset")
    print("2. Initialize model and tokenizer")
    print("3. Setup trainer with evaluation metrics")
    print("4. Train model with loss tracking")
    print("5. Evaluate on validation set")
    print("6. Save best model")

    print("\nNote: To run actual training, execute the main script:")
    print("python text_summarizer_main.py")

def demo_evaluation_metrics():
    """Demo evaluation metrics computation"""
    print("\n" + "="*60)
    print("DEMO: Evaluation Metrics")
    print("="*60)

    # Sample predictions and references for demo
    sample_predictions = [
        "AI is transforming technology and industries with language models like GPT-3.",
        "Climate change is a pressing challenge requiring immediate action to reduce emissions.",
        "The research shows significant improvements in model performance."
    ]

    sample_references = [
        "Artificial Intelligence has become transformative with models like GPT-3 reshaping industries.",
        "Climate change represents a major challenge that needs urgent action on greenhouse gas emissions.",
        "Research demonstrates substantial improvements in the performance of the models."
    ]

    evaluator = SummarizationEvaluator()
    metrics = evaluator.compute_all_metrics(sample_predictions, sample_references)
    evaluator.print_metrics(metrics)

def main():
    """Run all demos"""
    print("="*60)
    print("TEXT SUMMARIZATION SYSTEM DEMO")
    print("="*60)

    try:
        demo_data_loading()
        demo_model_comparison()
        demo_evaluation_metrics()
        demo_training_workflow()
        demo_inference()

        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext Steps:")
        print("1. Run 'python text_summarizer_main.py' to train a model")
        print("2. Use 'inference.py' for text summarization")
        print("3. Use 'evaluate.py' for model evaluation")
        print("4. Check 'data_loader.py' for dataset utilities")

    except Exception as e:
        logger.error(f"Demo error: {e}")

if __name__ == "__main__":
    main()
