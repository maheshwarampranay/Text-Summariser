
"""
Data Loading and Preprocessing Utilities for Text Summarization
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

class SummarizationDataset(Dataset):
    """Custom dataset for text summarization"""

    def __init__(self, texts, summaries, tokenizer, max_input_length=512, max_target_length=128):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        summary = str(self.summaries[idx])

        # Tokenize input text
        input_encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt'
        )

        # Tokenize target summary
        target_encodings = self.tokenizer(
            summary,
            truncation=True,
            padding='max_length',
            max_length=self.max_target_length,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encodings['input_ids'].flatten(),
            'attention_mask': input_encodings['attention_mask'].flatten(),
            'labels': target_encodings['input_ids'].flatten()
        }

class DatasetLoader:
    """Utility class for loading and preprocessing various summarization datasets"""

    SUPPORTED_DATASETS = {
        'cnn_dailymail': {
            'source_column': 'article',
            'target_column': 'highlights',
            'version': '3.0.0'
        },
        'xsum': {
            'source_column': 'document',
            'target_column': 'summary',
            'version': None
        },
        'billsum': {
            'source_column': 'text',
            'target_column': 'summary',
            'version': None
        }
    }

    @classmethod
    def load_dataset(cls, dataset_name, subset_size=None):
        """Load and return dataset"""
        if dataset_name not in cls.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(cls.SUPPORTED_DATASETS.keys())}")

        config = cls.SUPPORTED_DATASETS[dataset_name]

        if config['version']:
            dataset = load_dataset(dataset_name, config['version'])
        else:
            dataset = load_dataset(dataset_name)

        # Extract source and target columns
        train_texts = dataset['train'][config['source_column']]
        train_summaries = dataset['train'][config['target_column']]

        if 'validation' in dataset:
            val_texts = dataset['validation'][config['source_column']]
            val_summaries = dataset['validation'][config['target_column']]
        else:
            # Split train data if no validation set
            split_idx = int(len(train_texts) * 0.9)
            val_texts = train_texts[split_idx:]
            val_summaries = train_summaries[split_idx:]
            train_texts = train_texts[:split_idx]
            train_summaries = train_summaries[:split_idx]

        # Subset data if specified
        if subset_size:
            train_texts = train_texts[:subset_size]
            train_summaries = train_summaries[:subset_size]
            val_texts = val_texts[:subset_size//10]
            val_summaries = val_summaries[:subset_size//10]

        return {
            'train': {'texts': train_texts, 'summaries': train_summaries},
            'validation': {'texts': val_texts, 'summaries': val_summaries}
        }

    @classmethod
    def get_dataset_info(cls, dataset_name):
        """Get information about a dataset"""
        if dataset_name not in cls.SUPPORTED_DATASETS:
            return None

        if dataset_name == 'cnn_dailymail':
            return {
                'description': 'CNN/DailyMail dataset with news articles and highlights',
                'source': 'News articles',
                'target': 'Article highlights/summaries',
                'size': '~300k articles',
                'avg_source_length': '784 words',
                'task_type': 'News summarization'
            }
        elif dataset_name == 'xsum':
            return {
                'description': 'BBC articles with single-sentence summaries',
                'source': 'BBC news articles',
                'target': 'Single-sentence summaries',
                'size': '~226k articles',
                'avg_source_length': '430 words',
                'task_type': 'Extreme summarization'
            }
        elif dataset_name == 'billsum':
            return {
                'description': 'US Congressional bill summaries',
                'source': 'Legislative bill text',
                'target': 'Bill summaries',
                'size': '~23k bills',
                'avg_source_length': '2000+ words',
                'task_type': 'Legislative summarization'
            }
