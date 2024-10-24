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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    def __init__(self, csv_file: str):
        logger.info(f"Loading dataset from {csv_file}")
        df = pd.read_csv(csv_file)
        self.eng_spa_pairs = list(zip(df['english'].values, df['spanish'].values))
        
        self.special_tokens = {
            '<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3
        }
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

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        eng_sentences, spa_sentences = zip(*batch)
        eng_lengths = torch.tensor([len(s) for s in eng_sentences])
        
        eng_padded = pad_sequence(eng_sentences, batch_first=True, padding_value=0)
        spa_padded = pad_sequence(spa_sentences, batch_first=True, padding_value=0)
        
        return (eng_padded, eng_lengths), spa_padded

    def convert_indices_to_tokens(self, indices: np.ndarray) -> str:
        if isinstance(indices, np.int64):
            indices = [indices]
        reverse_vocab = {idx: word for word, idx in self.spa_vocab.items()}
        return ' '.join([reverse_vocab[idx] for idx in indices 
                        if idx not in (self.special_tokens['<sos>'], 
                                     self.special_tokens['<eos>'], 
                                     self.special_tokens['<pad>'])])

    

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
        
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(x)
        
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        # Process bidirectional hidden and cell states
        hidden = torch.cat([hidden[0:self.num_layers], hidden[self.num_layers:]], dim=2)
        cell = torch.cat([cell[0:self.num_layers], cell[self.num_layers:]], dim=2)
        
        hidden = self.fc_hidden(hidden)
        cell = self.fc_cell(cell)
        
        logger.debug(f"Encoder output shapes - outputs: {outputs.shape}, hidden: {hidden.shape}, cell: {cell.shape}")
        return outputs, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size)
        self.v = nn.Parameter(torch.rand(dec_hidden_size))
        
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # Repeat decoder hidden state src_len times
        # hidden: [batch_size, dec_hidden]
        # encoder_outputs: [batch_size, src_len, enc_hidden * 2]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        logger.debug(f"Attention shapes - hidden: {hidden.shape}, encoder_outputs: {encoder_outputs.shape}")
        
        # Calculate attention energies
        energy = torch.tanh(self.attention(
            torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, dec_hidden]
        energy = energy.permute(0, 2, 1)  # [batch_size, dec_hidden, src_len]
        
        # Calculate attention weights
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, dec_hidden]
        attention = torch.bmm(v, energy)  # [batch_size, 1, src_len]
        
        return torch.softmax(attention, dim=2)

class Decoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        if use_attention:
            self.attention = Attention(hidden_size * 2, hidden_size)
            lstm_input_size = hidden_size + hidden_size * 2
        else:
            lstm_input_size = hidden_size
            
        self.lstm = nn.LSTM(
            lstm_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        if use_attention:
            self.fc_out = nn.Linear(hidden_size + hidden_size * 2 + hidden_size, output_size)
        else:
            self.fc_out = nn.Linear(hidden_size, output_size)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], 
                encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input = input.unsqueeze(1)  # Add sequence length dimension
        embedded = self.dropout(self.embedding(input))
        
        if self.use_attention:
            # Get attention weights using the last layer's hidden state
            attention_weights = self.attention(hidden[0][-1], encoder_outputs)
            context = torch.bmm(attention_weights, encoder_outputs)
            rnn_input = torch.cat((embedded, context), dim=2)
        else:
            rnn_input = embedded
            
        logger.debug(f"Decoder shapes - input: {input.shape}, embedded: {embedded.shape}")
        if self.use_attention:
            logger.debug(f"Attention shapes - weights: {attention_weights.shape}, context: {context.shape}")
            
        output, hidden = self.lstm(rnn_input, hidden)
        
        if self.use_attention:
            output = torch.cat((output, context, embedded), dim=2)
            
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

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
        
        # First input to the decoder is the <sos> token
        decoder_input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t] = output
            
            # Teacher forcing
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
        src, src_lengths = batch[0]
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        trg = batch[1].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
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

def create_model(config: Dict, dataset: TranslationDataset, hidden_size: int, device: torch.device) -> Seq2SeqModel:
    encoder = Encoder(
        len(dataset.eng_vocab),
        hidden_size,
        num_layers=2,
        dropout=0.1
    )
    decoder = Decoder(
        len(dataset.spa_vocab),
        hidden_size,
        num_layers=2,
        dropout=0.1,
        use_attention=config['use_attention']
    )
    return Seq2SeqModel(encoder, decoder, device).to(device)
def evaluate_model(model: Seq2SeqModel, dataset: TranslationDataset, device: torch.device, 
                  config: Dict, batch_size: int = 32) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    all_predictions = []
    all_references = []
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            src, src_lengths = batch[0]
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            trg = batch[1].to(device)
            
            output = model(src, trg, src_lengths, teacher_forcing_ratio=0.0)
            
            # Calculate loss
            output_for_loss = output[:, 1:].reshape(-1, output.shape[-1])
            trg_for_loss = trg[:, 1:].reshape(-1)
            loss = criterion(output_for_loss, trg_for_loss)
            total_loss += loss.item()
            
            # Get predictions for BLEU score calculation
            predictions = output.argmax(dim=-1)  # [batch_size, max_len]
            
            # Convert predictions and references to text
            for pred_seq, ref_seq in zip(predictions, trg):
                pred_seq = pred_seq.cpu().numpy()
                ref_seq = ref_seq.cpu().numpy()
                
                # Convert prediction sequence to text
                pred_text = dataset.convert_indices_to_tokens(pred_seq)
                ref_text = dataset.convert_indices_to_tokens(ref_seq)
                
                all_predictions.append(pred_text)
                all_references.append([ref_text])  # sacrebleu expects a list of references for each prediction
    
    avg_loss = total_loss / len(data_loader)
    bleu = sacrebleu.corpus_bleu(all_predictions, all_references).score
    
    return bleu, avg_loss

@dataclass
class EvaluationMetrics:
    bleu_score: float
    loss: float
    training_time: float

class ResultManager:
    def __init__(self):
        self.results = {}

    def add_result(self, config_name: str, bleu: float, loss: float, training_time: float):
        self.results[config_name] = EvaluationMetrics(
            bleu_score=bleu,
            loss=loss,
            training_time=training_time
        )

    def save_results(self) -> None:
        # Convert results to DataFrame
        df = pd.DataFrame([
            {
                'Configuration': config_name,
                'BLEU Score': metrics.bleu_score,
                'Loss': metrics.loss,
                'Training Time (s)': metrics.training_time
            }
            for config_name, metrics in self.results.items()
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


# Update the train_and_evaluate function
def train_and_evaluate(config: Dict, dataset: TranslationDataset, device: torch.device, 
                      result_manager: ResultManager, num_epochs: int = 10, batch_size: int = 32) -> None:
    model = create_model(config, dataset, hidden_size=256, device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = amp.GradScaler()
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            clip=1.0,
            scaler=scaler, 
            device=device,
            teacher_forcing_ratio=0.5 if config['teacher_forcing'] else 0.0
        )
        if (epoch + 1) % 2 == 0:  
            bleu, eval_loss = evaluate_model(model, dataset, device, config, batch_size)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Eval Loss: {eval_loss:.4f} - BLEU: {bleu:.2f}")
    
    training_time = time.time() - start_time
    final_bleu, final_loss = evaluate_model(model, dataset, device, config, batch_size)
    
    # Add results to the result manager
    result_manager.add_result(config['name'], final_bleu, final_loss, training_time)

# Update the main block
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TranslationDataset("data1.csv")
    result_manager = ResultManager()
    
    configs = [
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
    
    num_epochs = 1
    for config in configs:
        print(f"Starting training with config: {config['name']}")
        train_and_evaluate(config, dataset, device, result_manager, num_epochs)
        print(f"Completed training with config: {config['name']}")
    
    # Save and visualize results
    result_manager.save_results()