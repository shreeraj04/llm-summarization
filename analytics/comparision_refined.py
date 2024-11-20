import os
import time
import torch
import logging
import psutil
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional
import argparse
from statistics import mean
import numpy as np
from rouge_score import rouge_scorer
import threading
import queue
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class CPUMonitor:
    def __init__(self, interval=0.5):
        self.process_cpu = []
        self.children_cpu = []
        self.system_cpu = []
        self.interval = interval
        self.stop_flag = False
        self.q = queue.Queue()
        self.num_cpus = psutil.cpu_count()

    def monitor(self):
        process = psutil.Process()
        while not self.stop_flag:
            try:
                # Get main process CPU
                main_process_cpu = process.cpu_percent()
                self.process_cpu.append(main_process_cpu)
                
                # Get child processes CPU
                children_cpu_total = 0
                for child in process.children(recursive=True):
                    try:
                        children_cpu_total += child.cpu_percent()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.children_cpu.append(children_cpu_total)
                
                # Get system-wide CPU
                system_cpu = psutil.cpu_percent(percpu=False)
                self.system_cpu.append(system_cpu)
                
                time.sleep(self.interval)
            except Exception as e:
                self.q.put(e)
                break

    def start(self):
        self.stop_flag = False
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.thread.join()
        
        try:
            exc = self.q.get_nowait()
            raise exc
        except queue.Empty:
            pass

        # Calculate statistics
        process_cpu_stats = {
            'avg': mean(self.process_cpu) if self.process_cpu else 0,
            'max': max(self.process_cpu) if self.process_cpu else 0,
            'min': min(self.process_cpu) if self.process_cpu else 0
        }
        
        children_cpu_stats = {
            'avg': mean(self.children_cpu) if self.children_cpu else 0,
            'max': max(self.children_cpu) if self.children_cpu else 0,
            'min': min(self.children_cpu) if self.children_cpu else 0
        }
        
        system_cpu_stats = {
            'avg': mean(self.system_cpu) if self.system_cpu else 0,
            'max': max(self.system_cpu) if self.system_cpu else 0,
            'min': min(self.system_cpu) if self.system_cpu else 0
        }
        
        total_process_avg = process_cpu_stats['avg'] + children_cpu_stats['avg']
        total_process_max = process_cpu_stats['max'] + children_cpu_stats['max']
        total_process_min = process_cpu_stats['min'] + children_cpu_stats['min']
        
        return {
            'main_process_cpu': process_cpu_stats,
            'children_cpu': children_cpu_stats,
            'total_process_cpu': {
                'avg': total_process_avg,
                'max': total_process_max,
                'min': total_process_min
            },
            'system_cpu': system_cpu_stats,
            'num_cpus': self.num_cpus
        }

@contextmanager
def cpu_monitoring():
    monitor = CPUMonitor()
    monitor.start()
    try:
        yield monitor
    finally:
        return monitor.stop()

class SummaryResult:
    def __init__(self, model_name, file_name, summary, processing_time, input_tokens, output_tokens,
                 memory_used_mb, cpu_stats, model_type, rouge_score):
        self.model_name = model_name
        self.file_name = file_name
        self.summary = summary
        self.processing_time = processing_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.memory_used_mb = memory_used_mb
        self.cpu_stats = cpu_stats  # Store the complete cpu_stats dictionary
        self.model_type = model_type
        self.rouge_score = rouge_score

class ConversationSummarizerCompare:
    def __init__(self, hf_token: str, device: Optional[str] = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.hf_token = hf_token
        logger.info(f"Using device: {self.device}")
        
        # Model configurations for different models
        self.model_configs = {
            'bart-conv': {
                'name': 'kabita-choudhary/finetuned-bart-for-conversation-summary',
                'prompt_template': "Summarise the conversation into few lines of paragraph. Here is the conversation: {conversation}",
                'type': 'conversation-specific'
            },
            'bart-azma': {
                'name': 'Azma-AI/bart-conversation-summarizer',
                'prompt_template': "Summarise the conversation into few lines of paragraph. Here is the conversation: {conversation}",
                'type': 'conversation-specific'
            },
            'bart-slauw': {
                'name': 'slauw87/bart_summarisation',
                'prompt_template': "Summarise the conversation into few lines of paragraph. Here is the conversation: {conversation}",
                'type': 'general'
            },
            'flan-t5': {
                'name': 'philschmid/flan-t5-base-samsum',
                'prompt_template': "Summarize the following dialogue into few lines of paragraph: {conversation}",
                'type': 'conversation-specific'
            },
            'dialogled': {
                'name': 'ConvAnalysis/DialogLED-base-16384-dialogsum-finetuned',
                'prompt_template': "Summarize this long conversation into few lines of paragraph: {conversation}",
                'type': 'conversation-specific'
            },
            # 'koalaai': {
            #     'name': 'KoalaAI/ChatSum-Large',
            #     'prompt_template': "Summarize the following conversation into few lines of paragraph: {conversation}",
            #     'type': 'conversation-specific'
            # }
        }

    def _load_model(self, model_name: str):
        """Loads model and tokenizer based on the model name"""
        try:
            model_name_or_path = self.model_configs[model_name]['name']
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
            return True
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def _get_memory_usage(self):
        """Returns the memory usage in MB"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB

    def _clear_memory(self):
        """Clear cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def summarize(self, conversation: str, model_name: str, file_name: str) -> SummaryResult:
        """Generate summary using the specified model"""
        if model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {model_name}")
        
        if not self._load_model(model_name):
            raise RuntimeError(f"Failed to load model: {model_name}")

        start_time = time.time()
        formatted_text = self.model_configs[model_name]['prompt_template'].format(conversation=conversation)
        initial_memory = self._get_memory_usage()
        cpu_stats = None
        summary = None
        input_tokens = 0
        output_tokens = 0
        summary_ids = None

        try:
            inputs = self.tokenizer(formatted_text, max_length=1024, truncation=True,return_tensors="pt").to(self.device)
            input_tokens = len(inputs["input_ids"][0])
            
            # Start CPU monitoring before model inference
            with cpu_monitoring() as monitor:
                summary_ids = self.model.generate(
                    inputs["input_ids"], 
                    max_length=200, 
                    min_length=40, 
                    length_penalty=1.5, 
                    num_beams=3, 
                    early_stopping=True,
                )
                cpu_stats = monitor.stop()  # This now returns a dictionary with CPU stats

            if summary_ids is not None:
                print(summary_ids)
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                output_tokens = len(summary_ids[0])

        except Exception as e:
            logger.error(f"Error generating summary with {model_name}: {str(e)}")
            raise
        finally:
            final_memory = self._get_memory_usage()
            memory_used = final_memory - initial_memory
            self._clear_memory()

        processing_time = time.time() - start_time

        if summary is None:
            raise RuntimeError(f"Failed to generate summary with {model_name}")

        # Calculate ROUGE score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True), summary)
        rouge_score = (rouge_scores['rouge1'].fmeasure + rouge_scores['rouge2'].fmeasure + rouge_scores['rougeL'].fmeasure) / 3

        return SummaryResult(
            model_name=model_name,
            file_name=file_name,
            summary=summary,
            processing_time=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            memory_used_mb=memory_used,
            cpu_stats=cpu_stats,  # Pass the complete cpu_stats dictionary
            model_type=self.model_configs[model_name]['type'],
            rouge_score=rouge_score
        )

def process_files(folder_path: str, hf_token: str) -> pd.DataFrame:
    """Process all text files in the folder and record results"""
    summarizer = ConversationSummarizerCompare(hf_token=hf_token)
    results = []
    
    # Get all text files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    total_files = len(files)
    
    for idx, file_name in enumerate(files, 1):
        file_path = os.path.join(folder_path, file_name)
        logger.info(f"\nProcessing file {idx}/{total_files}: {file_name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                conversation = file.read().strip()
            
            for model_name in ['bart-conv', 'bart-azma', 'bart-slauw', 'flan-t5', 'dialogled', 'koalaai']:
                logger.info(f"Processing with {model_name}...")
                try:
                    result = summarizer.summarize(conversation, model_name, file_name)
                    results.append({
                        'File': file_name,
                        'Model': model_name,
                        'Summary': result.summary,
                        'Processing Time (s)': result.processing_time,
                        'Memory Used (MB)': result.memory_used_mb,
                        'Main Process CPU Avg (%)': result.cpu_stats['main_process_cpu']['avg'],
                        'Main Process CPU Max (%)': result.cpu_stats['main_process_cpu']['max'],
                        'Main Process CPU Min (%)': result.cpu_stats['main_process_cpu']['min'],
                        'Children CPU Avg (%)': result.cpu_stats['children_cpu']['avg'],
                        'Children CPU Max (%)': result.cpu_stats['children_cpu']['max'],
                        'Children CPU Min (%)': result.cpu_stats['children_cpu']['min'],
                        'Total Process CPU Avg (%)': result.cpu_stats['total_process_cpu']['avg'],
                        'Total Process CPU Max (%)': result.cpu_stats['total_process_cpu']['max'],
                        'Total Process CPU Min (%)': result.cpu_stats['total_process_cpu']['min'],
                        'System CPU Avg (%)': result.cpu_stats['system_cpu']['avg'],
                        'System CPU Max (%)': result.cpu_stats['system_cpu']['max'],
                        'System CPU Min (%)': result.cpu_stats['system_cpu']['min'],
                        'Number of CPUs': result.cpu_stats['num_cpus'],
                        'Input Tokens': result.input_tokens,
                        'Output Tokens': result.output_tokens,
                        'Model Type': result.model_type,
                        'ROUGE Score': result.rouge_score
                    })
                except Exception as e:
                    logger.error(f"Failed to process {file_name} with {model_name}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\nDetailed CPU Usage and Performance Metrics:")
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        print(f"\n{'-'*50}")
        print(f"Model: {model}")
        print(f"{'-'*50}")
        
        # Processing time statistics
        print("\nProcessing Time Statistics:")
        print(f"  P99: {np.percentile(model_df['Processing Time (s)'], 99):.2f}s")
        print(f"  P95: {np.percentile(model_df['Processing Time (s)'], 95):.2f}s")
        print(f"  Average: {model_df['Processing Time (s)'].mean():.2f}s")
        
        # CPU Usage Statistics
        print("\nCPU Usage Statistics:")
        print("Process CPU (Including Children):")
        print(f"  Average: {model_df['Total Process CPU Avg (%)'].mean():.2f}%")
        print(f"  Maximum: {model_df['Total Process CPU Max (%)'].max():.2f}%")
        print(f"  Minimum: {model_df['Total Process CPU Min (%)'].min():.2f}%")
        
        print("\nSystem CPU:")
        print(f"  Average: {model_df['System CPU Avg (%)'].mean():.2f}%")
        print(f"  Maximum: {model_df['System CPU Max (%)'].max():.2f}%")
        print(f"  Minimum: {model_df['System CPU Min (%)'].min():.2f}%")
        
        # Memory Usage
        print(f"\nAverage Memory Used: {model_df['Memory Used (MB)'].mean():.2f} MB")
        
        # Model Performance
        print(f"Average ROUGE Score: {model_df['ROUGE Score'].mean():.4f}")
        
        print(f"\nNumber of CPUs: {model_df['Number of CPUs'].iloc[0]}")
    
    return df

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Summarize conversations using various models')
    parser.add_argument('--input', '-i', required=True, help='Path to folder containing conversation files')
    parser.add_argument('--output', '-o', required=True, help='Path to save results (CSV or XLSX)')
    parser.add_argument('--token', '-t', required=True, help='HuggingFace token')
    args = parser.parse_args()
    
    # Process all files and get results
    print("\nStarting file processing...")
    results_df = process_files(args.input, args.token)
    
    # Save results
    output_path = args.output
    if output_path.endswith('.xlsx'):
        results_df.to_excel(output_path, index=False)
    else:
        results_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")