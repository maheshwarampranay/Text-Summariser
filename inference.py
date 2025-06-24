
"""
Inference pipeline for text summarization
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging
from typing import List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TextSummarizerInference:
    """Easy-to-use text summarization inference pipeline"""

    def __init__(self, model_path: str = None, model_name: str = "t5-small"):
        """
        Initialize the summarizer

        Args:
            model_path: Path to fine-tuned model (if available)
            model_name: Pre-trained model name (used if model_path is None)
        """
        self.model_path = model_path or model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        self._load_model()

    def _load_model(self):
        """Load model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

            # Move model to device
            self.model.to(self.device)

            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == 'cuda' else -1
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def summarize(
        self,
        text: Union[str, List[str]],
        max_length: int = 130,
        min_length: int = 30,
        length_penalty: float = 2.0,
        num_beams: int = 4,
        early_stopping: bool = True,
        do_sample: bool = False,
        temperature: float = 1.0
    ) -> Union[str, List[str]]:
        """
        Generate summary for input text(s)

        Args:
            text: Input text or list of texts to summarize
            max_length: Maximum length of generated summary
            min_length: Minimum length of generated summary
            length_penalty: Length penalty for beam search
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early in beam search
            do_sample: Whether to use sampling
            temperature: Temperature for sampling

        Returns:
            Generated summary or list of summaries
        """
        try:
            # Handle single text input
            if isinstance(text, str):
                return self._summarize_single(
                    text, max_length, min_length, length_penalty,
                    num_beams, early_stopping, do_sample, temperature
                )

            # Handle batch input
            elif isinstance(text, list):
                return self._summarize_batch(
                    text, max_length, min_length, length_penalty,
                    num_beams, early_stopping, do_sample, temperature
                )

            else:
                raise ValueError("Input must be string or list of strings")

        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            raise

    def _summarize_single(self, text: str, max_length: int, min_length: int,
                         length_penalty: float, num_beams: int, early_stopping: bool,
                         do_sample: bool, temperature: float) -> str:
        """Summarize single text"""
        # Use pipeline for single text
        result = self.pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            early_stopping=early_stopping,
            do_sample=do_sample,
            temperature=temperature
        )

        return result[0]['summary_text']

    def _summarize_batch(self, texts: List[str], max_length: int, min_length: int,
                        length_penalty: float, num_beams: int, early_stopping: bool,
                        do_sample: bool, temperature: float) -> List[str]:
        """Summarize batch of texts"""
        summaries = []

        for text in texts:
            summary = self._summarize_single(
                text, max_length, min_length, length_penalty,
                num_beams, early_stopping, do_sample, temperature
            )
            summaries.append(summary)

        return summaries

    def summarize_with_custom_generation(
        self,
        text: str,
        max_new_tokens: int = 100,
        **generation_kwargs
    ) -> str:
        """
        Summarize with custom generation parameters

        Args:
            text: Input text
            max_new_tokens: Maximum number of new tokens to generate
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated summary
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)

        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )

        # Decode and return summary
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'model_type': self.model.__class__.__name__,
            'tokenizer_type': self.tokenizer.__class__.__name__,
            'vocab_size': self.tokenizer.vocab_size,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }

class BatchSummarizer:
    """Efficient batch processing for large datasets"""

    def __init__(self, model_path: str, batch_size: int = 8):
        self.summarizer = TextSummarizerInference(model_path)
        self.batch_size = batch_size

    def summarize_file(self, input_file: str, output_file: str, **generation_kwargs):
        """
        Summarize texts from file and save to output file

        Args:
            input_file: Path to input file with texts (one per line)
            output_file: Path to output file for summaries
            **generation_kwargs: Generation parameters
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]

            logger.info(f"Processing {len(texts)} texts from {input_file}")

            summaries = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_summaries = [
                    self.summarizer.summarize(text, **generation_kwargs) 
                    for text in batch
                ]
                summaries.extend(batch_summaries)

                logger.info(f"Processed batch {i//self.batch_size + 1}/{len(texts)//self.batch_size + 1}")

            # Save summaries
            with open(output_file, 'w', encoding='utf-8') as f:
                for summary in summaries:
                    f.write(summary + '\n')

            logger.info(f"Summaries saved to {output_file}")

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise

# Example usage functions
def demo_summarization():
    """Demo function showing how to use the summarizer"""
    # Sample text
    sample_text = """
    The Transformer architecture has revolutionized natural language processing since its introduction in 2017. 
    Unlike previous architectures that relied on recurrent or convolutional layers, Transformers use a mechanism 
    called attention to capture dependencies between words regardless of their distance in the sequence. This 
    allows for better parallelization during training and has led to significant improvements in performance 
    across various NLP tasks. The key innovation is the self-attention mechanism, which allows each position 
    in the sequence to attend to all positions in the previous layer. This has enabled the development of 
    large language models like BERT, GPT, and T5, which have achieved state-of-the-art results on numerous 
    benchmarks. The impact of Transformers extends beyond NLP to other domains like computer vision and 
    speech recognition, making it one of the most influential architectural innovations in deep learning.
    """

    # Initialize summarizer
    summarizer = TextSummarizerInference(model_name="t5-small")

    # Generate summary
    summary = summarizer.summarize(
        sample_text,
        max_length=100,
        min_length=30,
        num_beams=4
    )

    print("Original Text:")
    print(sample_text)
    print("\nGenerated Summary:")
    print(summary)

    # Show model info
    info = summarizer.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    demo_summarization()
