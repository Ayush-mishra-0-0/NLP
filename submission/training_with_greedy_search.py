import json
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm
import torch.cuda.amp as amp
from torch.nn.parallel import DataParallel
import numpy as np
from typing import Tuple, List, Dict
import logging
import time
import sacrebleu


# Add this class at the top of the file, after the imports
class Beam:
    def __init__(self, beam_size: int, vocab: dict, device: torch.device):
        self.beam_size = beam_size
        self.vocab = vocab
        self.device = device
        
        # Initialize with <sos> token
        self.tokens = torch.tensor([[vocab['<sos>']]], device=device)
        self.scores = torch.zeros(1, device=device)
        self.finished = []
        self.finished_scores = []
        
    def get_current_state(self) -> torch.Tensor:
        """Return current token sequences"""
        return self.tokens
    
    def advance(self, log_probs: torch.Tensor) -> None:
        """Update beam state based on new probabilities"""
        vocab_size = log_probs.size(-1)
        
        # Add current scores to new word scores
        scores = log_probs + self.scores.unsqueeze(1)
        
        # Flatten scores for all possible next tokens
        flat_scores = scores.view(-1)
        
        # Get top k scores and their indices
        best_scores, best_indices = flat_scores.topk(self.beam_size, 0, True, True)
        
        # Convert indices to word indices and beam indices
        beam_indices = best_indices // vocab_size
        word_indices = best_indices % vocab_size
        
        # Create new beam tokens
        new_tokens = torch.cat([
            self.tokens[beam_indices],
            word_indices.unsqueeze(1)
        ], dim=1)
        
        # Update beam state
        self.tokens = []
        self.scores = []
        
        for token_seq, score, word_idx in zip(new_tokens, best_scores, word_indices):
            if word_idx == self.vocab['<eos>']:
                # If sequence ended, add to finished sequences
                self.finished.append(token_seq)
                self.finished_scores.append(score)
            else:
                # Otherwise, keep in beam
                self.tokens.append(token_seq)
                self.scores.append(score)
                
        # If beam is empty or we have enough finished sequences
        if not self.tokens or len(self.finished) >= self.beam_size:
            self.tokens = []
            self.scores = []
        else:
            # Convert lists back to tensors
            self.tokens = torch.stack(self.tokens)
            self.scores = torch.stack(self.scores)
    
    def done(self) -> bool:
        """Check if beam search is complete"""
        return len(self.tokens) == 0
    
    def get_hyp(self) -> List[int]:
        """Get best hypothesis"""
        if self.finished:
            # Get hypothesis with best score
            best_idx = torch.tensor(self.finished_scores).argmax()
            return self.finished[best_idx].tolist()
        else:
            # If no finished sequences, return current best sequence
            return self.tokens[0].tolist()

# Rest of the code remains the same...

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    def __init__(self, csv_file: str):
        logger.info(f"Loading dataset from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Convert to numpy arrays for faster processing
        self.eng_spa_pairs = list(zip(df['english'].values, df['spanish'].values))
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3
        }
        
        # Create vocabularies with vectorized operations
        self.eng_vocab = self._create_vocabulary(df['english'].values)
        self.spa_vocab = self._create_vocabulary(df['spanish'].values)
        
        logger.info(f"Vocabulary sizes - English: {len(self.eng_vocab)}, Spanish: {len(self.spa_vocab)}")
    
    def _create_vocabulary(self, sentences: np.ndarray) -> Dict[str, int]:
        vocab = self.special_tokens.copy()
        word_id = len(self.special_tokens)
        
        words = set()
        for sentence in sentences:
            words.update(sentence.split())
        
        for word in words:
            if word not in vocab:
                vocab[word] = word_id
                word_id += 1
        return vocab
    
    def sentence_to_indices(self, sentence: str, vocab: Dict[str, int]) -> List[int]:
        return [vocab['<sos>']] + [vocab.get(word, vocab['<unk>']) 
                for word in sentence.split()] + [vocab['<eos>']]
    
    def __len__(self) -> int:
        return len(self.eng_spa_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        eng_sentence, spa_sentence = self.eng_spa_pairs[idx]
        return (
            torch.tensor(self.sentence_to_indices(eng_sentence, self.eng_vocab)),
            torch.tensor(self.sentence_to_indices(spa_sentence, self.spa_vocab))
        )

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    eng_sentences, spa_sentences = zip(*batch)
    eng_lengths = torch.tensor([len(s) for s in eng_sentences], dtype=torch.int64)
    
    eng_padded = pad_sequence(eng_sentences, batch_first=True, padding_value=0)
    spa_padded = pad_sequence(spa_sentences, batch_first=True, padding_value=0)
    
    return (eng_padded, eng_lengths), spa_padded



class Decoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = Attention(hidden_size)
        
        # Modified input size to match concatenated embedding and context vector
        self.lstm = nn.LSTM(
            hidden_size * 3,  # hidden_size (embedding) + hidden_size * 2 (encoder outputs)
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # Modified to account for concatenated context
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], 
                encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.unsqueeze(1)
        embedded = self.embedding(x)  # [batch_size, 1, hidden_size]
        
        # Get attention weights and context
        attention_weights = self.attention(hidden[0][-1].unsqueeze(1), encoder_outputs)
        context = torch.bmm(attention_weights, encoder_outputs)  # [batch_size, 1, hidden_size * 2]
        
        # Concatenate embedding with context vector
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, hidden_size * 3]
        
        # Pass through LSTM
        output, hidden = self.lstm(rnn_input, hidden)
        
        # Prepare output
        output = torch.cat((output, context), dim=2)
        predictions = self.out(output.squeeze(1))
        
        return predictions, hidden

class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Modified to output correct hidden size for decoder
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(x)
        lengths = lengths.cpu().to(torch.int64)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        # Process hidden and cell states for each layer
        hidden_processed = []
        cell_processed = []
        
        for layer in range(self.num_layers):
            # Combine forward and backward states for this layer
            hidden_cat = torch.cat((hidden[2*layer], hidden[2*layer+1]), dim=1)
            cell_cat = torch.cat((cell[2*layer], cell[2*layer+1]), dim=1)
            
            # Transform to decoder hidden size
            hidden_processed.append(self.fc_hidden(hidden_cat))
            cell_processed.append(self.fc_cell(cell_cat))
        
        # Stack processed states
        hidden = torch.stack(hidden_processed)
        cell = torch.stack(cell_processed)
        
        return outputs, (hidden, cell)
class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)  # Modified for bidirectional
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        hidden = hidden.repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy.transpose(1, 2))
        
        return self.dropout(torch.softmax(attention, dim=2))

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src: torch.Tensor, trg: torch.Tensor, src_lengths: torch.Tensor, 
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        decoder_input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = trg[:, t] if teacher_force else output.argmax(1)
        
        return outputs



def train_epoch(model: nn.Module, train_iterator: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, clip: float, scaler: amp.GradScaler, device: torch.device,
                teacher_forcing_ratio: float = 0.5) -> float:
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(train_iterator, desc='Training', leave=False)
    for batch in progress_bar:
        # Move batch to device
        (src, src_lengths), trg = batch
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Use torch.amp.autocast() instead of cuda.amp.autocast()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = model(src, trg, src_lengths, teacher_forcing_ratio)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return epoch_loss / len(train_iterator)

@dataclass
class EvaluationMetrics:
    bleu_score: float
    loss: float
    training_time: float
    
class ConfigurationEvaluator:
    def __init__(
        self,
        dataset: TranslationDataset,
        hidden_size: int,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        device: torch.device
    ):
        self.dataset = dataset
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.results = {}
        
    def create_model(self, config: Dict) -> Seq2SeqModel:
        encoder = Encoder(
            len(self.dataset.eng_vocab),
            self.hidden_size,
            num_layers=2,
            dropout=0.1
        )
        
        decoder = Decoder(
            len(self.dataset.spa_vocab),
            self.hidden_size,
            num_layers=2,
            dropout=0.1
        )
        
        model = Seq2SeqModel(encoder, decoder, self.device)
        return model.to(self.device)
    
    def beam_search_decode(
        self,
        model: Seq2SeqModel,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        beam_size: int = 5,
        max_len: int = 50
    ) -> List[List[int]]:
        model.eval()
        batch_size = src.size(0)
        
        # Encoder forward pass
        encoder_outputs, hidden = model.encoder(src, src_lengths)
        
        # Initialize beam for each batch item
        beams = [Beam(beam_size, self.dataset.spa_vocab, self.device) for _ in range(batch_size)]
        
        # Process each time step
        for _ in range(max_len):
            all_candidates = []
            for beam in beams:
                if beam.done():
                    continue
                    
                # Get current tokens and scores
                tokens = beam.get_current_state()
                decoder_input = tokens[:, -1].unsqueeze(1)
                
                # Decoder forward pass
                with torch.no_grad():
                    output, hidden = model.decoder(
                        decoder_input,
                        hidden,
                        encoder_outputs
                    )
                
                log_probs = torch.log_softmax(output, dim=-1)
                beam.advance(log_probs)
            
            # Break if all beams are done
            if all(beam.done() for beam in beams):
                break
        
        # Get best hypotheses
        all_hyps = []
        for beam in beams:
            hyps = beam.get_hyp()
            all_hyps.append(hyps)
            
        return all_hyps
    
    def greedy_decode(
        self,
        model: Seq2SeqModel,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_len: int = 50
    ) -> List[List[int]]:
        model.eval()
        batch_size = src.size(0)
        
        # Encoder forward pass
        encoder_outputs, hidden = model.encoder(src, src_lengths)
        
        # Initialize decoder input
        decoder_input = torch.tensor([self.dataset.spa_vocab['<sos>']] * batch_size).to(self.device)
        
        outputs = []
        for _ in range(max_len):
            # Decoder forward pass
            with torch.no_grad():
                output, hidden = model.decoder(
                    decoder_input,
                    hidden,
                    encoder_outputs
                )
            
            # Get most likely next token
            decoder_input = output.argmax(1)
            outputs.append(decoder_input.unsqueeze(1))
            
            # Break if all sequences have ended
            if all(token.item() == self.dataset.spa_vocab['<eos>'] for token in decoder_input):
                break
        
        return torch.cat(outputs, dim=1).tolist()
    
    def calculate_metrics(
        self,
        hypotheses: List[List[int]],
        references: List[List[int]]
    ) -> EvaluationMetrics:
        # Convert indices back to words
        idx_to_spa = {v: k for k, v in self.dataset.spa_vocab.items()}
        
        hyp_sentences = [
            ' '.join([idx_to_spa[idx] for idx in hyp if idx not in [0, 1, 2]])
            for hyp in hypotheses
        ]
        
        ref_sentences = [
            ' '.join([idx_to_spa[idx] for idx in ref if idx not in [0, 1, 2]])
            for ref in references
        ]
        
        # Calculate BLEU score
        bleu_score = sacrebleu.corpus_bleu(hyp_sentences, [ref_sentences]).score
        
        return bleu_score
    
    def evaluate_configuration(self, config: Dict) -> EvaluationMetrics:
        # Create data loaders
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model and training components
        model = self.create_model(config)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        scaler = torch.cuda.amp.GradScaler()
        
        # Training
        start_time = time.time()
        for epoch in range(self.num_epochs):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                clip=1.0,
                scaler=scaler,
                device=self.device,
                teacher_forcing_ratio=0.5 if config['teacher_forcing'] else 0.0
            )
        
        training_time = time.time() - start_time
        
        # Evaluation
        model.eval()
        test_translations = []
        test_references = []
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc='Evaluating'):
                src, src_lengths = batch[0]
                src = src.to(self.device)
                src_lengths = src_lengths.to(self.device)
                
                if config['use_beam_search']:
                    translations = self.beam_search_decode(model, src, src_lengths)
                else:
                    translations = self.greedy_decode(model, src, src_lengths)
                
                test_translations.extend(translations)
                test_references.extend(batch[1].tolist())
        
        bleu_score = self.calculate_metrics(test_translations, test_references)
        
        return EvaluationMetrics(
            bleu_score=bleu_score,
            loss=train_loss,
            training_time=training_time
        )
    
    def evaluate_all_configurations(self) -> Dict[str, EvaluationMetrics]:
        configurations = [
            {
                'use_attention': False,
                'use_beam_search': False,
                'teacher_forcing': True,
                'name': 'No attention, greedy, with TF'
            },
             {
                'use_attention': True,
                'use_beam_search': False,
                'teacher_forcing': False,
                'name': 'Attention, greedy, without TF'
            },
            {
                'use_attention': True,
                'use_beam_search': True,
                'teacher_forcing': True,
                'name': 'Attention, beam search, with TF'
            },
           
            {
                'use_attention': False,
                'use_beam_search': True,
                'teacher_forcing': False,
                'name': 'No attention, beam search, without TF'
            }
        ]
        
        results = {}
        for config in tqdm(configurations, desc='Evaluating configurations'):
            metrics = self.evaluate_configuration(config)
            results[config['name']] = metrics
            
            # Save intermediate results
            self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, EvaluationMetrics]) -> None:
        # Convert results to DataFrame
        df = pd.DataFrame([
            {
                'Configuration': config_name,
                'BLEU Score': metrics.bleu_score,
                'Loss': metrics.loss,
                'Training Time (s)': metrics.training_time
            }
            for config_name, metrics in results.items()
        ])
        
        # Save as CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(f'evaluation_results_{timestamp}.csv', index=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # BLEU scores
        plt.subplot(1, 2, 1)
        plt.bar(df['Configuration'], df['BLEU Score'])
        plt.xticks(rotation=45)
        plt.title('BLEU Scores by Configuration')
        
        # Training time
        plt.subplot(1, 2, 2)
        plt.bar(df['Configuration'], df['Training Time (s)'])
        plt.xticks(rotation=45)
        plt.title('Training Time by Configuration')
        
        plt.tight_layout()
        plt.savefig(f'evaluation_results_{timestamp}.png')
        plt.close()

# Usage example
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = TranslationDataset("data1.csv")
    
    # Initialize evaluator
    evaluator = ConfigurationEvaluator(
        dataset=dataset,
        hidden_size=512,
        learning_rate=3e-4,
        batch_size=128,
        num_epochs=1,
        device=device
    )
    
    # Run evaluation
    results = evaluator.evaluate_all_configurations()
    
    # Print final results
    print("\nFinal Results:")
    for config_name, metrics in results.items():
        print(f"\n{config_name}:")
        print(f"BLEU Score: {metrics.bleu_score:.2f}")
        print(f"Final Loss: {metrics.loss:.4f}")
        print(f"Training Time: {metrics.training_time:.2f}s")

if __name__ == "__main__":
    main()