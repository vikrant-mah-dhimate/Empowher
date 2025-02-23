import torch
import logging
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return model, tokenizer

def mask_names_and_orgs(text, ner_model, ner_tokenizer, id2label):
    """
    Masks names (PER) and organizations (ORG) in the given text using the provided model and tokenizer.
    """
    # Tokenize input text
    tokens = ner_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = ner_model(**tokens)

    # Get predictions
    predictions = torch.argmax(outputs.logits, dim=2)
    tokenized_text = ner_tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    # Log tokenized text and predictions (excluding CLS and SEP tokens)
    logger.debug(f"Original Tokenized Text: {tokenized_text}")
    
    # Remove the special tokens `[CLS]` and `[SEP]`
    special_tokens = ['[CLS]', '[SEP]']
    tokenized_text = [token for token in tokenized_text if token not in special_tokens]
    predictions = predictions[0][1:-1]  # Remove predictions for [CLS] and [SEP]

    # Log the cleaned tokens and predictions
    logger.debug(f"Cleaned Tokenized Text: {tokenized_text}")
    logger.debug(f"Predictions: {[id2label[pred.item()] for pred in predictions]}")

    # Create a list to track entity spans
    entity_spans_dict = {}
    current_entity = None
    span_start_idx = None

    # Iterate through the tokenized text and predictions
    for idx, (token, pred) in enumerate(zip(tokenized_text, predictions)):
        entity_label = id2label[pred.item()]

        # Start of a new entity
        if entity_label == "B-PER" or entity_label == "I-PER":
            if current_entity != "PER":
                if current_entity is not None:
                    # End of previous entity, save span (only store the first token for the span)
                    for i in range(span_start_idx, idx):
                        entity_spans_dict[i] = [current_entity]
                span_start_idx = idx
                current_entity = "PER"
       #elif entity_label == "B-ORG" or entity_label == "I-ORG":
       #    if current_entity != "ORG":
       #        if current_entity is not None:
       #            # End of previous entity, save span (only store the first token for the span)
       #            for i in range(span_start_idx, idx):
       #                entity_spans_dict[i] = [current_entity]
       #        span_start_idx = idx
       #        current_entity = "ORG"
        else:
            if current_entity is not None:
                # End of previous entity, save span (only store the first token for the span)
                for i in range(span_start_idx, idx):
                    entity_spans_dict[i] = [current_entity]
            current_entity = None
            span_start_idx = None

    # Handle last entity span
    if current_entity is not None:
        for i in range(span_start_idx, len(tokenized_text)):
            entity_spans_dict[i] = [current_entity]

    # Mask entities in tokenized text (only mask the first token of the span)
    masked_tokens = tokenized_text[:]
    for idx, entity_types in entity_spans_dict.items():
        # Mask only the first token of the entity span
        if "PER" in entity_types:
            mask_token = "[MASKED_PERSON]"
        elif "ORG" in entity_types:
            mask_token = "[MASKED_ORG]"
        else:
            continue

        # Replace the token at the start of the span with the mask token
        masked_tokens[idx] = mask_token

    # Reconstruct the masked text
    masked_text = ner_tokenizer.convert_tokens_to_string(masked_tokens)

    return masked_text


