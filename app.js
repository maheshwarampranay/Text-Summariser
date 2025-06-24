// Text Summarizer Application JavaScript

class TextSummarizer {
    constructor() {
        this.models = [
            {"name": "T5-Small", "rouge1": 35.4, "rouge2": 14.0, "rougeL": 28.5, "meteor": 35.0, "description": "Best for abstractive summarization"},
            {"name": "BART", "rouge1": 30.8, "rouge2": 15.7, "rougeL": 26.2, "meteor": 28.0, "description": "Strong performance on news articles"},
            {"name": "PEGASUS", "rouge1": 24.5, "rouge2": 12.0, "rougeL": 21.8, "meteor": 25.0, "description": "Pre-trained for summarization"}
        ];
        
        this.examples = [
            {
                "title": "Artificial Intelligence",
                "text": "Artificial Intelligence has become one of the most transformative technologies of the 21st century. From machine learning algorithms that power recommendation systems to deep learning models that can generate human-like text, AI is reshaping industries and changing how we interact with technology. The field has seen rapid advancement in recent years, particularly with the development of large language models like GPT-3 and BERT. These models have demonstrated remarkable capabilities in understanding and generating natural language, leading to applications in chatbots, content creation, and automated customer service. However, the development of AI also raises important ethical considerations, including concerns about bias, privacy, and the potential impact on employment. As AI continues to evolve, it's crucial that we develop frameworks for responsible AI deployment and ensure that these powerful technologies benefit society as a whole.",
                "summaries": {
                    "T5-Small": "AI has become transformative in the 21st century, with language models like GPT-3 and BERT advancing natural language capabilities. While enabling applications in chatbots and content creation, AI development raises ethical concerns about bias, privacy, and employment impact, requiring responsible deployment frameworks.",
                    "BART": "Artificial Intelligence is reshaping industries through machine learning and deep learning models. Large language models demonstrate remarkable natural language capabilities but raise ethical considerations including bias and privacy concerns that need addressing for responsible AI deployment.",
                    "PEGASUS": "AI technologies like machine learning and language models are transforming industries. The development of models like GPT-3 enables new applications but creates ethical challenges requiring responsible frameworks."
                }
            },
            {
                "title": "Climate Change",
                "text": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are leading to significant environmental changes worldwide. These include melting polar ice caps, rising sea levels, more frequent extreme weather events, and shifts in precipitation patterns. The impacts extend beyond environmental concerns to affect human health, food security, water resources, and economic stability. Scientists have reached a strong consensus that immediate action is needed to reduce greenhouse gas emissions and transition to renewable energy sources. Governments, businesses, and individuals all have important roles to play in addressing this challenge through policy changes, technological innovation, and behavioral modifications. The next decade is considered critical for implementing effective climate action to limit global warming and mitigate its most severe consequences.",
                "summaries": {
                    "T5-Small": "Climate change poses pressing challenges through rising temperatures and greenhouse gas emissions, causing environmental changes like melting ice caps and extreme weather. Scientists emphasize immediate action through renewable energy transition, with governments, businesses, and individuals playing crucial roles in the critical next decade.",
                    "BART": "Climate change from greenhouse gas emissions causes rising temperatures, melting ice caps, and extreme weather affecting health, food security, and economy. Scientists agree on immediate action needed for renewable energy transition and emission reduction through collective efforts.",
                    "PEGASUS": "Rising global temperatures from greenhouse gases cause environmental changes including melting ice and extreme weather. Immediate action needed for renewable energy transition to address this pressing challenge."
                }
            },
            {
                "title": "Transformer Architecture",
                "text": "The Transformer architecture has revolutionized natural language processing since its introduction in 2017. Unlike previous architectures that relied on recurrent or convolutional layers, Transformers use a mechanism called attention to capture dependencies between words regardless of their distance in the sequence. This allows for better parallelization during training and has led to significant improvements in performance across various NLP tasks. The key innovation is the self-attention mechanism, which allows each position in the sequence to attend to all positions in the previous layer. This has enabled the development of large language models like BERT, GPT, and T5, which have achieved state-of-the-art results on numerous benchmarks. The impact of Transformers extends beyond NLP to other domains like computer vision and speech recognition, making it one of the most influential architectural innovations in deep learning.",
                "summaries": {
                    "T5-Small": "Transformer architecture revolutionized NLP in 2017 using attention mechanisms instead of recurrent layers, enabling better parallelization and performance. The self-attention mechanism led to models like BERT, GPT, and T5, with impact extending beyond NLP to computer vision and speech recognition.",
                    "BART": "Transformers revolutionized NLP through attention mechanisms that capture word dependencies regardless of distance, enabling parallelization and improved performance. This led to successful models like BERT and GPT with applications beyond NLP.",
                    "PEGASUS": "Transformer architecture uses attention mechanisms for better NLP performance, enabling models like BERT and GPT. The innovation extends to computer vision and speech recognition domains."
                }
            }
        ];
        
        this.summaryLengths = [
            { label: "Short", range: "50-80 words", factor: 0.6 },
            { label: "Medium", range: "80-120 words", factor: 1.0 },
            { label: "Long", range: "120-200 words", factor: 1.4 }
        ];
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.updateModelDescription();
        this.updateSummaryLengthDescription();
        this.updateAdvancedValues();
        this.updateWordCount();
        this.highlightSelectedModel();
    }
    
    bindEvents() {
        // Example buttons
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const exampleIndex = parseInt(e.currentTarget.dataset.example);
                this.loadExample(exampleIndex);
            });
        });
        
        // Model selection
        document.getElementById('modelSelect').addEventListener('change', () => {
            this.updateModelDescription();
            this.highlightSelectedModel();
        });
        
        // Summary length slider
        document.getElementById('summaryLength').addEventListener('input', () => {
            this.updateSummaryLengthDescription();
        });
        
        // Advanced settings sliders
        document.getElementById('beamCount').addEventListener('input', () => {
            this.updateAdvancedValues();
        });
        
        document.getElementById('temperature').addEventListener('input', () => {
            this.updateAdvancedValues();
        });
        
        // Input text area
        document.getElementById('inputText').addEventListener('input', () => {
            this.updateWordCount();
        });
        
        // Summarize button
        document.getElementById('summarizeBtn').addEventListener('click', () => {
            this.summarizeText();
        });
    }
    
    loadExample(index) {
        const example = this.examples[index];
        document.getElementById('inputText').value = example.text;
        this.updateWordCount();
        
        // Add visual feedback
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.classList.remove('btn--primary');
            btn.classList.add('btn--outline');
        });
        
        document.querySelectorAll('.example-btn')[index].classList.remove('btn--outline');
        document.querySelectorAll('.example-btn')[index].classList.add('btn--primary');
        
        setTimeout(() => {
            document.querySelectorAll('.example-btn')[index].classList.remove('btn--primary');
            document.querySelectorAll('.example-btn')[index].classList.add('btn--outline');
        }, 1000);
    }
    
    updateModelDescription() {
        const selectedModel = document.getElementById('modelSelect').value;
        const model = this.models.find(m => m.name === selectedModel);
        document.getElementById('modelDescription').textContent = model.description;
    }
    
    updateSummaryLengthDescription() {
        const value = parseInt(document.getElementById('summaryLength').value);
        const lengthInfo = this.summaryLengths[value];
        document.getElementById('summaryLengthDesc').textContent = lengthInfo.range;
    }
    
    updateAdvancedValues() {
        const beamValue = document.getElementById('beamCount').value;
        const tempValue = parseFloat(document.getElementById('temperature').value).toFixed(1);
        
        document.getElementById('beamValue').textContent = beamValue;
        document.getElementById('tempValue').textContent = tempValue;
    }
    
    updateWordCount() {
        const text = document.getElementById('inputText').value;
        const wordCount = this.countWords(text);
        document.getElementById('inputWordCount').textContent = `${wordCount} words`;
    }
    
    highlightSelectedModel() {
        const selectedModel = document.getElementById('modelSelect').value;
        
        document.querySelectorAll('.model-row').forEach(row => {
            row.classList.remove('active');
            if (row.dataset.model === selectedModel) {
                row.classList.add('active');
            }
        });
    }
    
    countWords(text) {
        return text.trim() === '' ? 0 : text.trim().split(/\s+/).length;
    }
    
    async summarizeText() {
        const inputText = document.getElementById('inputText').value.trim();
        
        if (!inputText) {
            alert('Please enter some text to summarize or load an example.');
            return;
        }
        
        if (this.countWords(inputText) < 50) {
            alert('Please enter at least 50 words for meaningful summarization.');
            return;
        }
        
        // Show loading state
        this.showLoadingState();
        
        // Simulate processing time
        const processingTime = Math.random() * 2 + 1; // 1-3 seconds
        
        await new Promise(resolve => setTimeout(resolve, processingTime * 1000));
        
        // Generate summary
        const summary = this.generateSummary(inputText);
        
        // Show results
        this.showResults(inputText, summary, processingTime);
        
        // Hide loading state
        this.hideLoadingState();
    }
    
    showLoadingState() {
        const btn = document.getElementById('summarizeBtn');
        const spinner = document.getElementById('loadingSpinner');
        
        btn.classList.add('loading');
        btn.disabled = true;
        spinner.classList.remove('hidden');
        
        // Show processing status
        const resultsCard = document.getElementById('resultsCard');
        resultsCard.style.display = 'block';
        
        const summaryOutput = document.getElementById('summaryOutput');
        summaryOutput.classList.add('loading');
        summaryOutput.textContent = 'Processing your text...';
    }
    
    hideLoadingState() {
        const btn = document.getElementById('summarizeBtn');
        const spinner = document.getElementById('loadingSpinner');
        
        btn.classList.remove('loading');
        btn.disabled = false;
        spinner.classList.add('hidden');
        
        const summaryOutput = document.getElementById('summaryOutput');
        summaryOutput.classList.remove('loading');
    }
    
    generateSummary(inputText) {
        const selectedModel = document.getElementById('modelSelect').value;
        const summaryLengthIndex = parseInt(document.getElementById('summaryLength').value);
        const lengthFactor = this.summaryLengths[summaryLengthIndex].factor;
        
        // Check if this is one of our example texts
        for (const example of this.examples) {
            if (inputText.includes(example.text.substring(0, 100))) {
                let baseSummary = example.summaries[selectedModel];
                return this.adjustSummaryLength(baseSummary, lengthFactor);
            }
        }
        
        // Generate a realistic summary for custom text
        return this.generateCustomSummary(inputText, selectedModel, lengthFactor);
    }
    
    adjustSummaryLength(baseSummary, factor) {
        const sentences = baseSummary.split('. ');
        
        if (factor < 0.8) {
            // Short summary - take first sentence or two
            return sentences.slice(0, Math.max(1, Math.floor(sentences.length * 0.6))).join('. ') + '.';
        } else if (factor > 1.2) {
            // Long summary - add more detail
            const expanded = baseSummary + ' This comprehensive analysis highlights the key aspects and implications discussed in the original text, providing valuable insights into the subject matter.';
            return expanded;
        }
        
        return baseSummary;
    }
    
    generateCustomSummary(inputText, model, lengthFactor) {
        // Extract key phrases and create a realistic summary
        const sentences = inputText.split('. ');
        const wordCount = this.countWords(inputText);
        
        let targetLength;
        if (lengthFactor < 0.8) {
            targetLength = Math.floor(wordCount * 0.15);
        } else if (lengthFactor > 1.2) {
            targetLength = Math.floor(wordCount * 0.35);
        } else {
            targetLength = Math.floor(wordCount * 0.25);
        }
        
        // Simple extractive summarization simulation
        const importantSentences = sentences
            .filter(s => s.length > 20)
            .slice(0, Math.max(2, Math.floor(sentences.length * 0.4)));
        
        let summary = importantSentences.join('. ') + '.';
        
        // Adjust based on model characteristics
        switch (model) {
            case 'T5-Small':
                summary = 'This analysis reveals that ' + summary.toLowerCase();
                break;
            case 'BART':
                summary = 'The key findings indicate that ' + summary.toLowerCase();
                break;
            case 'PEGASUS':
                summary = 'In summary, ' + summary.toLowerCase();
                break;
        }
        
        return summary;
    }
    
    showResults(originalText, summary, processingTime) {
        const originalWords = this.countWords(originalText);
        const summaryWords = this.countWords(summary);
        const compressionRatio = Math.round(originalWords / summaryWords * 10) / 10;
        
        // Update summary output
        const summaryOutput = document.getElementById('summaryOutput');
        summaryOutput.textContent = summary;
        summaryOutput.classList.add('results-appear');
        
        // Update statistics
        document.getElementById('originalWords').textContent = originalWords;
        document.getElementById('summaryWords').textContent = summaryWords;
        document.getElementById('compressionRatio').textContent = `${compressionRatio}:1`;
        document.getElementById('processingTime').textContent = `${processingTime.toFixed(1)}s`;
        
        // Show results card
        const resultsCard = document.getElementById('resultsCard');
        resultsCard.style.display = 'block';
        resultsCard.classList.add('results-appear');
        
        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Remove animation class after animation completes
        setTimeout(() => {
            summaryOutput.classList.remove('results-appear');
            resultsCard.classList.remove('results-appear');
        }, 300);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TextSummarizer();
});

// Add some utility functions for enhanced UX
document.addEventListener('DOMContentLoaded', () => {
    // Add smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to summarize
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            document.getElementById('summarizeBtn').click();
        }
        
        // Escape to clear input
        if (e.key === 'Escape' && document.activeElement === document.getElementById('inputText')) {
            if (confirm('Clear the input text?')) {
                document.getElementById('inputText').value = '';
                document.getElementById('inputWordCount').textContent = '0 words';
            }
        }
    });
    
    // Add copy to clipboard functionality for summary
    document.addEventListener('click', (e) => {
        if (e.target.id === 'summaryOutput' && e.target.textContent.trim()) {
            navigator.clipboard.writeText(e.target.textContent).then(() => {
                // Show feedback
                const originalText = e.target.textContent;
                e.target.textContent = 'Summary copied to clipboard!';
                e.target.style.color = 'var(--color-success)';
                
                setTimeout(() => {
                    e.target.textContent = originalText;
                    e.target.style.color = '';
                }, 1500);
            }).catch(() => {
                console.log('Copy to clipboard failed');
            });
        }
    });
    
    // Add tooltips for model performance metrics
    const tooltips = {
        'ROUGE-1': 'Measures overlap of unigrams between summary and reference',
        'ROUGE-2': 'Measures overlap of bigrams between summary and reference', 
        'ROUGE-L': 'Measures longest common subsequence between summary and reference',
        'METEOR': 'Considers precision, recall, and semantic similarity'
    };
    
    document.querySelectorAll('th').forEach(th => {
        if (tooltips[th.textContent]) {
            th.title = tooltips[th.textContent];
            th.style.cursor = 'help';
        }
    });
});