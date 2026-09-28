from typing import List
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor  # Added for parallelization
import random
import threading
import nltk
from nltk.corpus import wordnet
import torch
from transformers import BertTokenizer, BertForMaskedLM

PARAPHRASE_PROMPT = (
    "Please rewrite the following text and return only the rewritten text: "
)


def parallel_edit(text_editor, texts, max_workers=128, desc="Paraphrasing"):
    """Paraphrase a list of texts in parallel.

    Parameters:
        text_editor: An object with an `.edit(text)` method.
        texts (List[str]): The texts to paraphrase.
        max_workers (int): Number of parallel worker threads.
        desc (str): Description for the tqdm progress bar.

    Returns:
        List[str]: The edited texts, in the original order.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            tqdm(executor.map(text_editor.edit, texts), total=len(texts), desc=desc)
        )


class TextEditor:
    def __init__(self):
        self.desc = "Editing"
        pass

    def edit(self, input: str) -> str:
        raise NotImplementedError

    def edit_batch(self, inputs: List[str]) -> List[str]:
        edited_outputs = parallel_edit(self, inputs, desc=self.desc)
        return edited_outputs


class TextParaphraser(TextEditor):
    def __init__(self, model_name: str, openai_url: str, api_key: str):
        client = OpenAI(
            base_url=openai_url,
            api_key=api_key,
        )
        self.client = client
        self.prompt = PARAPHRASE_PROMPT
        self.model_name = model_name
        self.desc = "Paraphrasing"

    def edit(self, input: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self.prompt + input}],
        )
        return response.choices[0].message.content


class TextBackTranslation(TextEditor):
    def __init__(
        self, model_name: str, openai_url: str, api_key: str, language: str = "French"
    ):
        client = OpenAI(
            base_url=openai_url,
            api_key=api_key,
        )
        self.client = client
        self.prompt = "Translate the following text to {language}. Your reply should only contain the translated text.\n\n"
        self.model_name = model_name
        self.language = language
        self.desc = f"Translating to {self.language} and back to English"

    def edit(self, input: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt.format(language=self.language) + input,
                }
            ],
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt.format(language="English")
                    + response.choices[0].message.content
                }
            ],
        )
        return response.choices[0].message.content

# Editors below are forked from: https://github.com/THU-BPM/MarkLLM/blob/main/evaluation/tools/text_editor.py
class WordDeletion(TextEditor):
    """Delete words randomly from the text."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the word deletion editor.

            Parameters:
                ratio (float): The ratio of words to delete.
        """
        self.ratio = ratio
        self.desc = f"Deleting words with ratio {self.ratio}"

    def edit(self, input: str) -> str:
        """Delete words randomly from the text."""

        # Handle empty string input
        if not input:  
            return input

        # Split the text into words and randomly delete each word based on the ratio
        word_list = input.split()
        edited_words = [word for word in word_list if random.random() >= self.ratio]

        # Join the words back into a single string
        deleted_text = ' '.join(edited_words)

        return deleted_text

class SynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        nltk.download('wordnet')
        wordnet.ensure_loaded()  # Preload to avoid LazyCorpusLoader races with threads
        self._wordnet_lock = threading.Lock()
        self._synonym_cache: dict[str, list] = {}
        self.desc = f"Substituting synonyms with ratio {self.ratio}"

    def _get_synonyms(self, word: str):
        """Thread-safe retrieval of synsets for a word, with caching."""
        cached = self._synonym_cache.get(word)
        if cached is not None:
            return cached

        with self._wordnet_lock:
            # Another thread may have filled the cache while we waited.
            if word in self._synonym_cache:
                return self._synonym_cache[word]
            synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
            self._synonym_cache[word] = synonyms
            return synonyms

    def edit(self, input: str) -> str:
        """Randomly replace words with synonyms from WordNet."""
        words = input.split()
        num_words = len(words)
        
        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                word_synonyms[word] = self._get_synonyms(word)
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)
        
            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text


class ContextAwareSynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet based on the context."""

    def __init__(self, ratio: float, tokenizer: BertTokenizer, model: BertForMaskedLM, device='cuda') -> None:
        """
        Initialize the context-aware synonym substitution editor.

        Parameters:
            ratio (float): The ratio of words to replace.
            tokenizer (BertTokenizer): Tokenizer for BERT model.
            model (BertForMaskedLM): BERT model for masked language modeling.
            device (str): Device to run the model (e.g., 'cuda', 'cpu').
        """
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        nltk.download('wordnet')
        self.desc = f"Context-aware substituting synonyms with ratio {self.ratio}"
    
    def _get_synonyms_from_wordnet(self, word: str):
        """ Return a list of synonyms for the given word using WordNet. """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def edit(self, input: str) -> str:
        """Randomly replace words with synonyms from WordNet based on the context."""
        words = input.split()
        num_words = len(words)
        replaceable_indices = []

        for i, word in enumerate(words):
            if self._get_synonyms_from_wordnet(word):
                replaceable_indices.append(i)

        num_to_replace = int(min(self.ratio, len(replaceable_indices) / num_words) * num_words)
        indices_to_replace = random.sample(replaceable_indices, num_to_replace)

        real_replace = 0

        for i in indices_to_replace:
            # Create a sentence with a [MASK] token
            masked_sentence = words[:i] + ['[MASK]'] + words[i+1:]
            masked_text = " ".join(masked_sentence)
            
            # Use BERT to predict the token for [MASK]
            inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0].item()

            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = outputs.logits[0, mask_position]
            predicted_indices = torch.argsort(predictions, descending=True)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
            words[i] = predicted_tokens[0]
            real_replace += 1
        
        replaced_text = ' '.join(words)

        return replaced_text
