import json
import random
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from pathlib import Path
import os
from phoneme_control import RhymeController
import logging
import torch.optim as optim
import math
from collections import Counter

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='/tmp/debug.log')

class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size, 
                             padding=padding, dilation=dilation)
        self.glu = nn.GLU(dim=1)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.glu(out)
        out = out[:, :, :-self.conv.padding[0]]
        residual = self.residual(x)
        out = (out + residual).transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        return out

class TCN(nn.Module):
    def __init__(self, vocab_size: int, channels: int = 256, layers: int = 5):
        super(TCN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, channels)
        self.tcn = nn.ModuleList([
            TCNBlock(channels, channels, kernel_size=3, dilation=2**i)
            for i in range(layers)
        ])
        self.fc = nn.Linear(channels, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        for tcn_block in self.tcn:
            x = tcn_block(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return x

class TCNLyricGenerator:
    def __init__(self, processed_data_path: str = "processed_data", model_path: str = "model.pt", max_vocab_size: int = 20000):
        print("Initializing TCNLyricGenerator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processed_data_path = Path(processed_data_path)
        self.model_path = Path(model_path)
        self.max_vocab_size = max_vocab_size
        self.lyrics_data = self._load_processed_data()
        self.word_to_tag = self._build_word_to_tag()
        self.rhyme_controller = RhymeController()
        self.rhyme_controller.lyrics_data = self.lyrics_data
        self.available_genres = self._get_available_genres()
        self.vocab, self.word2idx, self.idx2word = self._build_vocab()
        self.model_config = {'vocab_size': len(self.vocab), 'channels': 256, 'layers': 5}
        self.model = None
        print(f"Vocab size: {len(self.vocab)}, Genres: {self.available_genres}")

    def _load_processed_data(self) -> List[Dict]:
        file_path = Path('/kaggle/input/jjjjjj/processed_lyrics.json')
        print(f"Loading data from {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid_data = []
        invalid_count = 0
        for i, item in enumerate(data):
            text = item.get('text', '')
            genre = item.get('genre', '')
            # Try multiple tag field names
            tags = (item.get('tags') or item.get('pos_tags') or item.get('POSTags') or [])
            if isinstance(text, str) and text.strip() and isinstance(genre, str) and genre.strip():
                valid_data.append({'text': text, 'genre': genre, 'tags': tags})
            else:
                invalid_count += 1
                logging.debug(f"Invalid entry {i}: text={text[:50]}, genre={genre}, tags={tags[:10]}")
        print(f"Loaded {len(data)} entries, {len(valid_data)} valid, {invalid_count} invalid")
        love_data = [item for item in valid_data if item['genre'].lower() == 'love']
        valid_data.extend(love_data)
        return valid_data

    def _build_word_to_tag(self) -> Dict[str, str]:
        word_tag_counts = Counter()
        for item in self.lyrics_data:
            text = item.get('text', '')
            tags = item.get('tags', [])
            if not isinstance(text, str) or not text.strip() or not isinstance(tags, list):
                logging.debug(f"Skipping item, text={text[:50]}, tags={tags}")
                continue
            words = text.strip().split()
            if len(words) != len(tags):
                logging.debug(f"Tag mismatch: {len(words)} words, {len(tags)} tags, text={text[:50]}")
                continue
            for word, tag in zip(words, tags):
                if isinstance(tag, str) and tag.strip():
                    word_tag_counts[(word.strip(), tag.upper())] += 1
                else:
                    logging.debug(f"Invalid tag for word={word}, tag={tag}")
        word_to_tag = {}
        for (word, tag), count in word_tag_counts.items():
            if word not in word_to_tag or word_tag_counts[(word, tag)] > word_tag_counts.get((word, word_to_tag[word]), 0):
                word_to_tag[word] = tag
        print(f"Built word_to_tag with {len(word_to_tag)} entries")
        logging.debug(f"Word_to_tag sample: {list(word_to_tag.items())[:10]}")
        return word_to_tag

    def _get_available_genres(self) -> List[str]:
        return sorted(set(item['genre'].strip().lower() for item in self.lyrics_data if item['genre']))

    def _tokenize_and_tag(self, text: str, item: Dict = None) -> List[Tuple[str, str]]:
        words = text.strip().split()
        if item and 'tags' in item and isinstance(item['tags'], list) and len(item['tags']) == len(words):
            return [(word.strip(), tag.upper()) for word, tag in zip(words, item['tags']) if isinstance(tag, str)]
        return [self._fallback_tag(word) for word in words]

    def _fallback_tag(self, word: str) -> Tuple[str, str]:
        tag = self.word_to_tag.get(word.strip(), 'OTHER')
        return (word.strip(), tag)

    def _build_vocab(self) -> Tuple[set, Dict[str, int], Dict[int, str]]:
        vocab = set(['காதல்[NOUN]', 'நிலா[NOUN]'])
        for item in self.lyrics_data:
            text = item.get('text', '')
            if not isinstance(text, str) or not text.strip():
                continue
            tokens = self._tokenize_and_tag(text, item)
            for word, tag in tokens:
                vocab.add(f"{word}[{tag}]")
        vocab.add('<PAD>')
        vocab.add('<UNK>')
        if len(vocab) > self.max_vocab_size:
            word_counts = Counter()
            for item in self.lyrics_data:
                tokens = self._tokenize_and_tag(item.get('text', ''), item)
                for word, tag in tokens:
                    word_counts[f"{word}[{tag}]"] += 1
            vocab = set(w for w, c in word_counts.most_common(self.max_vocab_size - 2))
            vocab.add('<PAD>')
            vocab.add('<UNK>')
        print("Vocabulary sample:", list(vocab)[:20])
        word2idx = {word: idx for idx, word in enumerate(sorted(vocab))}
        idx2word = {idx: word for word, idx in word2idx.items()}
        return vocab, word2idx, idx2word

    def _prepare_data(self, sequence_length: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        for item in self.lyrics_data:
            text = item.get('text', '')
            tokens = self._tokenize_and_tag(text, item)
            token_ids = [self.word2idx.get(f"{word}[{tag}]", self.word2idx['<UNK>']) for word, tag in tokens]
            for i in range(0, len(token_ids) - sequence_length, 1):
                sequences.append(token_ids[i:i + sequence_length + 1])
        print(f"Prepared {len(sequences)} sequences")
        sequences = torch.tensor(sequences, dtype=torch.long).to(self.device)
        train_size = int(0.9 * len(sequences))
        train_data = sequences[:train_size]
        val_data = sequences[train_size:]
        print(f"Train sequences: {len(train_data)}, Validation sequences: {len(val_data)}")
        return train_data, val_data

    def _compute_perplexity(self, data: torch.Tensor, batch_size: int = 16) -> float:
        self.model.eval()
        total_loss = 0
        total_count = 0
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:min(i + batch_size, len(data))]
                inputs, targets = batch[:, :-1], batch[:, 1:]
                try:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.view(-1, len(self.vocab)), targets.reshape(-1))
                    if not torch.isinf(loss) and not torch.isnan(loss):
                        total_loss += loss.item() * inputs.size(0)
                        total_count += inputs.size(0)
                except Exception as e:
                    logging.debug(f"Perplexity error: {str(e)}")
                    continue
        avg_loss = total_loss / total_count if total_count > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss != float('inf') else float('inf')
        self.model.train()
        return perplexity

    def train_and_save(self, genre: str, keywords: List[str], num_lines: int, rhyme_scheme: str, epochs: int = 15):
        print("Starting TCN training...")
        self.model = TCN(**self.model_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.word2idx['<PAD>'])
        valid_keywords = [k for k in keywords if f"{k}[NOUN]" in self.vocab]
        if len(valid_keywords) < len(keywords):
            print(f"Warning: Keywords not in vocab: {set(keywords) - set(valid_keywords)}")
        self.model.train()
        train_data, val_data = self._prepare_data(sequence_length=30)
        batch_size = 16
        best_perplexity = float('inf')
        patience = 5
        patience_counter = 0
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:min(i + batch_size, len(train_data))]
                inputs, targets = batch[:, :-1], batch[:, 1:]
                self.optimizer.zero_grad()
                try:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.view(-1, len(self.vocab)), targets.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    total_loss += loss.item()
                    batch_count += 1
                except Exception as e:
                    logging.debug(f"Training batch {i//batch_size} error: {str(e)}")
                    continue
            if batch_count == 0:
                print(f"Epoch {epoch+1}: No valid batches processed")
                continue
            avg_loss = total_loss / batch_count
            train_perplexity = self._compute_perplexity(train_data, batch_size)
            val_perplexity = self._compute_perplexity(val_data, batch_size)
            try:
                sample_lyrics = self._generate_lyrics_with_model(num_lines, max_line_length=5, keywords=keywords)
                rhyme_accuracy = 100.0 if self.rhyme_controller.validate_rhyme_scheme(sample_lyrics, rhyme_scheme) else 0.0
                keyword_inclusion = (sum(1 for k in keywords if any(k in line for line in sample_lyrics)) /
                                    len(keywords) * 100) if keywords else 100.0
            except Exception as e:
                print(f"Error in lyric generation at epoch {epoch+1}: {str(e)}")
                sample_lyrics = ["Error in generation"]
                rhyme_accuracy = 0.0
                keyword_inclusion = 0.0
            print(f"Epoch {epoch+1}/{epochs} ({(epoch+1)/epochs*100:.2f}% complete) | "
                  f"Avg Loss: {avg_loss:.4f} | Train Perplexity: {train_perplexity:.2f} | "
                  f"Val Perplexity: {val_perplexity:.2f}")
            print(f"Sample Lyrics: {sample_lyrics}")
            print(f"Metrics: Rhyme Accuracy: {rhyme_accuracy:.1f}%, Keyword Inclusion: {keyword_inclusion:.1f}%")
            if val_perplexity < best_perplexity:
                best_perplexity = val_perplexity
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}, best perplexity: {best_perplexity:.2f}")
                    break
        print(f"Training complete, model saved to {self.model_path}")

    def _generate_lyrics_with_model(self, num_lines: int, max_line_length: int, keywords: List[str] = None) -> List[str]:
        if self.model is None:
            raise ValueError("Model not loaded")
        self.model.eval()
        lyrics = []
        valid_keywords = [k for k in (keywords or []) if f"{k}[NOUN]" in self.vocab]
        pos_pattern = ['NOUN', 'NOUN', 'VERB', 'ADJ']
        used_keywords = []
        for line_num in range(num_lines):
            for attempt in range(5):
                line = []
                target_pos = pos_pattern[line_num % len(pos_pattern)]
                available_keywords = [k for k in valid_keywords if k not in used_keywords]
                if not available_keywords:
                    available_keywords = valid_keywords
                start_word = random.choice(available_keywords) if available_keywords else random.choice(list(self.vocab - {'<PAD>', '<UNK>'}))
                input_ids = [self.word2idx.get(start_word, self.word2idx['<UNK>'])]
                input_tensor = torch.tensor([input_ids[-10:]], dtype=torch.long).to(self.device)
                line.append(start_word.split('[')[0] if '[' in start_word else start_word)
                for _ in range(max_line_length - 1):
                    with torch.no_grad():
                        output = self.model(input_tensor)
                    probs = torch.softmax(output[:, -1, :], dim=-1)
                    next_idx = torch.multinomial(probs, 1).item()
                    next_word = self.idx2word[next_idx]
                    if next_word in ['<PAD>', '<UNK>']:
                        continue
                    try:
                        word, tag = next_word.split('[')
                        tag = tag.rstrip(']')
                    except ValueError:
                        continue
                    if tag != target_pos and random.random() > 0.3:
                        continue
                    word = word.strip()
                    line.append(word)
                    input_ids.append(next_idx)
                    input_tensor = torch.tensor([input_ids[-10:]], dtype=torch.long).to(self.device)
                    if len(line) >= 3 and ('.' in word or len(line) == max_line_length):
                        break
                line_text = " ".join(line).strip()
                if line_text and len(line) >= 3:
                    lyrics.append(line_text)
                    if any(k in line_text for k in valid_keywords):
                        used_keywords.append(start_word.split('[')[0] if '[' in start_word else start_word)
                    break
            else:
                line_text = random.choice(valid_keywords) if valid_keywords else "காதல்"
                lyrics.append(f"{line_text} மழை தூறும்")
        self.model.train()
        return lyrics[:num_lines]

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"No trained model found at {self.model_path}")
        self.model = TCN(**self.model_config).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {self.model_path}")

    def generate_lyrics(self, genre: str = None, keywords: List[str] = None, 
                       num_lines: int = 4, rhyme_scheme: str = "AABB", 
                       max_line_length: int = 5) -> Tuple[List[str], Dict]:
        print(f"Generating lyrics: genre={genre}, keywords={keywords}")
        if self.model is None:
            self.load_model()
        lyrics = self._generate_lyrics_with_model(num_lines, max_line_length, keywords)
        metrics = {}
        metrics['rhyme_accuracy'] = 100.0 if self.rhyme_controller.validate_rhyme_scheme(lyrics, rhyme_scheme) else 0.0
        metrics['keyword_inclusion'] = (sum(1 for k in (keywords or []) if any(k in line for line in lyrics)) /
                                      len(keywords) * 100) if keywords else 100.0
        print(f"Generated lyrics: {lyrics}")
        print(f"Metrics: {metrics}")
        return lyrics, metrics