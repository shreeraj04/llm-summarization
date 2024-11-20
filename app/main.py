from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, validator
from typing import Optional, List, Dict
import torch
import logging
import psutil
import time
import os
from pathlib import Path
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer
)
import spacy
from keybert import KeyBERT
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Conversation Summary API")

class ConversationInput(BaseModel):
    conversation_id: str
    sender: Optional[str] = None
    model_name: str
    conversation_text: Optional[str] = None
    file_path: Optional[str] = None

    @validator('file_path')
    def validate_file_path(cls, v):
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"File not found: {v}")
            if not path.is_file():
                raise ValueError(f"Path is not a file: {v}")
            return str(path.absolute())
        return v

    @validator('conversation_text', 'file_path')
    def validate_input(cls, v, values):
        if 'conversation_text' in values and 'file_path' in values:
            if values['conversation_text'] is None and values['file_path'] is None:
                raise ValueError("Either conversation_text or file_path must be provided")
        return v

class SummaryResponse(BaseModel):
    conversation_id: str
    sender: Optional[str]
    model_name: str
    summary: str
    keywords: List[str]
    named_entities: Dict[str, List[str]]
    performance_metrics: Dict[str, float]
    file_path: Optional[str] = None

class ModelManager:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}
        self.tokenizers = {}
        self.nlp = spacy.load("en_core_web_sm")
        self.keyword_model = KeyBERT()
        
        # Model configurations
        self.model_configs = {
            'bart-conv': {
                'name': 'kabita-choudhary/finetuned-bart-for-conversation-summary',
                'prompt_template': "Summarise the conversation into few lines of paragraph. Here is the conversation: {conversation}"
            },
            'bart-azma': {
                'name': 'Azma-AI/bart-conversation-summarizer',
                'prompt_template': "Summarise the conversation into few lines of paragraph. Here is the conversation: {conversation}"
            },
            'flan-t5': {
                'name': 'philschmid/flan-t5-base-samsum',
                'prompt_template': "Summarize the following dialogue into few lines of paragraph: {conversation}"
            },
            'dialogled': {
                'name': 'ConvAnalysis/DialogLED-base-16384-dialogsum-finetuned',
                'prompt_template': "Summarize this long conversation into few lines of paragraph: {conversation}"
            }
        }

    def load_models(self):
        """Load all models during startup"""
        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Loading model: {model_name}")
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(config['name'])
                self.models[model_name] = AutoModelForSeq2SeqLM.from_pretrained(config['name']).to(self.device)
                logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")

    def get_memory_usage(self):
        """Returns the memory usage in MB"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text using KeyBERT"""
        keywords = self.keyword_model.extract_keywords(text, top_n=top_n)
        return [keyword for keyword, _ in keywords]

    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities

    def summarize(self, conversation: str, model_name: str) -> Dict:
        """Generate summary and analytics"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        start_time = time.time()
        initial_memory = self.get_memory_usage()

        # Format conversation with prompt template
        formatted_text = self.model_configs[model_name]['prompt_template'].format(
            conversation=conversation
        )

        # Generate summary
        inputs = self.tokenizers[model_name](
            formatted_text, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        summary_ids = self.models[model_name].generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.tokenizers[model_name].decode(summary_ids[0], skip_special_tokens=True)

        # Extract keywords and named entities
        keywords = self.extract_keywords(conversation)
        named_entities = self.extract_named_entities(conversation)

        # Calculate performance metrics
        processing_time = time.time() - start_time
        memory_used = self.get_memory_usage() - initial_memory

        performance_metrics = {
            'processing_time_seconds': processing_time,
            'memory_used_mb': memory_used,
            'input_tokens': len(inputs["input_ids"][0]),
            'output_tokens': len(summary_ids[0])
        }

        return {
            'summary': summary,
            'keywords': keywords,
            'named_entities': named_entities,
            'performance_metrics': performance_metrics
        }

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load models during startup"""
    model_manager.load_models()

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_conversation(conversation_input: ConversationInput):
    try:
        # Get conversation text either from direct input or file
        if conversation_input.conversation_text:
            conversation_text = conversation_input.conversation_text
        elif conversation_input.file_path:
            try:
                with open(conversation_input.file_path, 'r', encoding='utf-8') as f:
                    conversation_text = f.read().strip()
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading file: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either conversation_text or file_path must be provided"
            )

        result = model_manager.summarize(
            conversation_text,
            conversation_input.model_name
        )

        return SummaryResponse(
            conversation_id=conversation_input.conversation_id,
            sender=conversation_input.sender,
            model_name=conversation_input.model_name,
            summary=result['summary'],
            keywords=result['keywords'],
            named_entities=result['named_entities'],
            performance_metrics=result['performance_metrics'],
            file_path=conversation_input.file_path
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.post("/summarize/upload", response_model=SummaryResponse)
async def summarize_uploaded_conversation(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    sender: Optional[str] = Form(None),
    model_name: str = Form(...)
):
    try:
        conversation_text = await file.read()
        conversation_text = conversation_text.decode('utf-8')

        result = model_manager.summarize(
            conversation_text,
            model_name
        )

        return SummaryResponse(
            conversation_id=conversation_id,
            sender=sender,
            model_name=model_name,
            summary=result['summary'],
            keywords=result['keywords'],
            named_entities=result['named_entities'],
            performance_metrics=result['performance_metrics']
        )

    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing uploaded file: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)