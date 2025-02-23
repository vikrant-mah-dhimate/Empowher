import torch
import logging
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class NamedEntityMasker:
    def __init__(self, model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
        """
        Initializes the NamedEntityMasker class.
        Loads the model and tokenizer for Named Entity Recognition (NER).

        Args:
        model_name (str): Pretrained model name for NER (default is "dbmdz/bert-large-cased-finetuned-conll03-english").
        """
        # Load model and tokenizer
        self.ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.id2label = self.ner_model.config.id2label

    def mask_names_and_orgs(self, text):
        """
        Masks names (PER) and organizations (ORG) in the given text using the NER model.

        Args:
        text (str): Input text to process for named entities.

        Returns:
        str: Text with masked entities.
        """
        # Tokenize input text
        tokens = self.ner_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.ner_model(**tokens)

        # Get predictions
        predictions = torch.argmax(outputs.logits, dim=2)
        tokenized_text = self.ner_tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

        # Log tokenized text and predictions (excluding CLS and SEP tokens)
        logger.debug(f"Original Tokenized Text: {tokenized_text}")

        # Remove special tokens [CLS] and [SEP]
        special_tokens = ['[CLS]', '[SEP]']
        tokenized_text = [token for token in tokenized_text if token not in special_tokens]
        predictions = predictions[0][1:-1]  # Remove predictions for [CLS] and [SEP]

        logger.debug(f"Cleaned Tokenized Text: {tokenized_text}")
        logger.debug(f"Predictions: {[self.id2label[pred.item()] for pred in predictions]}")

        # Track entity spans (store only the first token of the entity)
        entity_spans_dict = {}
        current_entity = None
        span_start_idx = None

        # Iterate through the tokenized text and predictions
        for idx, (token, pred) in enumerate(zip(tokenized_text, predictions)):
            entity_label = self.id2label[pred.item()]

            # Start of a new entity
            if entity_label == "B-PER" or entity_label == "I-PER":
                if current_entity != "PER":
                    if current_entity is not None:
                        # End of previous entity, save span (store only the first token)
                        for i in range(span_start_idx, idx):
                            entity_spans_dict[i] = [current_entity]
                    span_start_idx = idx
                    current_entity = "PER"
            elif entity_label == "B-ORG" or entity_label == "I-ORG":
                if current_entity != "ORG":
                    if current_entity is not None:
                        # End of previous entity, save span (store only the first token)
                        for i in range(span_start_idx, idx):
                            entity_spans_dict[i] = [current_entity]
                    span_start_idx = idx
                    current_entity = "ORG"
            else:
                if current_entity is not None:
                    # End of previous entity, save span (store only the first token)
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
        
        # Keep track of whether a mask was already applied
        entity_first_token_masked = {}

        for idx, entity_types in entity_spans_dict.items():
            # Mask only the first token of the entity span
            if "PER" in entity_types:
                mask_token = "[MASKED_PERSON]"
            elif "ORG" in entity_types:
                mask_token = "[MASKED_ORG]"
            else:
                continue

            # Apply mask only to the first token of the entity span
            if idx not in entity_first_token_masked:
                masked_tokens[idx] = mask_token
                entity_first_token_masked[idx] = True
            else:
                # Replace all other tokens in the span with an empty space
                masked_tokens[idx] = " "

        # Reconstruct the masked text
        masked_text = self.ner_tokenizer.convert_tokens_to_string(masked_tokens)

        return masked_text


# Example Usage
if __name__ == "__main__":
    # Instantiate the NamedEntityMasker
    masker = NamedEntityMasker()

    # Input text
    text = "Vikrant is in trouble. Google and Microsoft are big tech companies."

    # Mask names and organizations in the text
    masked_text = masker.mask_names_and_orgs(text)
    
    # Output the masked text
    print(f"Masked Text: {masked_text}")

