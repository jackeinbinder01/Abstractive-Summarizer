from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch


def make_bart_pipeline():
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokenizer.model_max_length = model.config.max_position_embeddings
    return pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        device=device,
    )


def make_pegasus_pipeline():
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available

    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
    tokenizer.model_max_length = model.config.max_position_embeddings
    return pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        device=device,
    )
