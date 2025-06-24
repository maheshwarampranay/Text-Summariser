
"""
Text Summarization Training Script
Similar to the style shown in the uploaded notebook
"""

import os
import torch
import logging
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the text summarization model"""
    model_name: str = "t5-small"  # Can be t5-small, facebook/bart-base, google/pegasus-xsum
    dataset_name: str = "cnn_dailymail"
    dataset_version: str = "3.0.0"
    max_input_length: int = 512
    max_target_length: int = 128
    output_dir: str = "./Coderone_2ndProject"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000

class TextSummarizer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.rouge_metric = evaluate.load("rouge")

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model and tokenizer loaded successfully")

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        logger.info(f"Loading {self.config.dataset_name} dataset")

        if self.config.dataset_name == "cnn_dailymail":
            dataset = load_dataset("cnn_dailymail", self.config.dataset_version)

            def preprocess_function(examples):
                # For CNN/DailyMail, use article as input and highlights as target
                inputs = [doc for doc in examples["article"]]
                targets = [doc for doc in examples["highlights"]]

                model_inputs = self.tokenizer(
                    inputs, 
                    max_length=self.config.max_input_length, 
                    truncation=True, 
                    padding=True
                )

                # Setup the tokenizer for targets
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        targets, 
                        max_length=self.config.max_target_length, 
                        truncation=True, 
                        padding=True
                    )

                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

        elif self.config.dataset_name == "xsum":
            dataset = load_dataset("xsum")

            def preprocess_function(examples):
                inputs = [doc for doc in examples["document"]]
                targets = [doc for doc in examples["summary"]]

                model_inputs = self.tokenizer(
                    inputs, 
                    max_length=self.config.max_input_length, 
                    truncation=True, 
                    padding=True
                )

                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        targets, 
                        max_length=self.config.max_target_length, 
                        truncation=True, 
                        padding=True
                    )

                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

        # Preprocess datasets
        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # Use smaller subsets for faster training (similar to the notebook style)
        train_dataset = tokenized_dataset["train"].select(range(1000))  # Small subset for demo
        eval_dataset = tokenized_dataset["validation"].select(range(100))

        return train_dataset, eval_dataset

    def compute_metrics(self, eval_pred):
        """Compute ROUGE metrics for evaluation"""
        predictions, labels = eval_pred

        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        result = self.rouge_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            use_stemmer=True
        )

        # Extract F1 scores
        result = {key: value * 100 for key, value in result.items()}

        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
            "rouge_score": result["rouge1"]  # Use ROUGE-1 as main metric
        }

    def setup_trainer(self, train_dataset, eval_dataset):
        """Setup the Hugging Face Trainer"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="rouge_score",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard for simplicity
            push_to_hub=False,
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=self.model, 
            padding=True
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        logger.info("Trainer setup complete")

    def train(self):
        """Train the model"""
        logger.info("Starting training...")

        # Train the model
        train_result = self.trainer.train()

        # Log training results (similar to the notebook style)
        logger.info("Training completed!")
        logger.info(f"Training Loss: {train_result.training_loss}")
        logger.info(f"Global Step: {train_result.global_step}")

        # Save the model (similar to the notebook: trainer.save_model('Coderone_2ndProject'))
        self.trainer.save_model(self.config.output_dir)
        logger.info(f"Model saved to {self.config.output_dir}")

        return train_result

    def evaluate(self):
        """Evaluate the model"""
        logger.info("Evaluating model...")
        eval_result = self.trainer.evaluate()

        logger.info("Evaluation Results:")
        for key, value in eval_result.items():
            logger.info(f"{key}: {value}")

        return eval_result

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """Generate summary for input text"""
        if self.model is None or self.tokenizer is None:
            # Load model for inference if not already loaded
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.output_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.output_dir)

        # Tokenize input
        inputs = self.tokenizer.encode(
            text, 
            return_tensors="pt", 
            max_length=self.config.max_input_length, 
            truncation=True
        )

        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

def main():
    """Main training function"""
    # Configuration
    config = ModelConfig(
        model_name="t5-small",  # Can change to facebook/bart-base, google/pegasus-xsum
        dataset_name="cnn_dailymail",
        num_train_epochs=1,  # Reduced for demo
        per_device_train_batch_size=4,  # Reduced for memory
        per_device_eval_batch_size=4,
    )

    # Initialize summarizer
    summarizer = TextSummarizer(config)

    # Setup model and tokenizer
    summarizer.setup_model_and_tokenizer()

    # Load and preprocess data
    train_dataset, eval_dataset = summarizer.load_and_preprocess_data()

    # Setup trainer
    summarizer.setup_trainer(train_dataset, eval_dataset)

    # Train model
    train_result = summarizer.train()

    # Evaluate model
    eval_result = summarizer.evaluate()

    # Test inference
    test_text = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
    Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially 
    criticized by some of France's leading artists and intellectuals for its design, but it 
    has become a global cultural icon of France and one of the most recognizable structures 
    in the world. The Eiffel Tower is the tallest structure in Paris and the second-tallest 
    free-standing structure in France after the Millau Viaduct.
    """

    summary = summarizer.summarize(test_text)
    print(f"\nOriginal Text: {test_text}")
    print(f"\nGenerated Summary: {summary}")

if __name__ == "__main__":
    main()
