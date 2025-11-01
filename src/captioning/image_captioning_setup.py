#!pip install efficientnet_pytorch
#!pip install editdistance  # For accurate CER calculation
#!pip install pycocoevalcap
#!pip install torch torchvision efficientnet-pytorch sklearn matplotlib tqdm pillow

import os
import json
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import editdistance

try:
    from pycocoevalcap.bleu.bleu import Bleu as CocoBleu
    from pycocoevalcap.cider.cider import Cider as CocoCider
    from pycocoevalcap.spice.spice import Spice as CocoSpice
    COCO_EVAL_AVAILABLE = True
except Exception:
    COCO_EVAL_AVAILABLE = False

try:
    from nltk.translate.bleu_score import corpus_bleu
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

#  Vocabulary
class Vocab:
    def __init__(self):
        self.token2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3, "<EMPTY>": 4}
        self.idx2token = {idx: tk for tk, idx in self.token2idx.items()}
        self.next_idx = 5

    def add_sentence(self, sentence):
        if sentence == "":
            sentence = '<EMPTY>'
        for ch in sentence:
            if ch not in self.token2idx:
                self.token2idx[ch] = self.next_idx
                self.idx2token[self.next_idx] = ch
                self.next_idx += 1

    def encode(self, sentence):
        if sentence == "":
            sentence = '<EMPTY>'
        tokens = [self.token2idx.get(ch, self.token2idx['<UNK>']) for ch in sentence]
        return [self.token2idx['<SOS>']] + tokens + [self.token2idx['<EOS>']]

    def decode(self, idxs):
        tokens = []
        for idx in idxs:
            tok = self.idx2token.get(idx, '<UNK>')
            if tok in ['<SOS>', '<EOS>', '<PAD>']:
                continue
            tokens.append(tok)
        return ''.join(tokens)

    def __len__(self):
        return len(self.token2idx)

#  Dataset
def load_entries(json_path):
    print(f"Loading entries from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    excluded_prefixes = {'002', '004', '009', '018', '020', '022', '024', '025', '027', '039', '040', '043'}
    return [e for e in data if e.get('alignment') == 'horizontal' and e['path'][:3] not in excluded_prefixes]

class BubbleDataset(Dataset):
    def __init__(self, entries, vocab, crop_dir, transform):
        self.transform = transform
        self.crop_root = crop_dir
        self.vocab = vocab
        self.flat = []

        for e in entries:
            n_forms = e.get('nforms', 0)
            answers = e.get('answer', [])

            for crop_idx in range(n_forms):
                if crop_idx < len(answers) and answers[crop_idx] and len(answers[crop_idx]) > 0:
                    answer_text = answers[crop_idx][0]
                    if answer_text:  
                        self.flat.append((e, crop_idx))

        print(f"Created dataset with {len(self.flat)} samples")

    def __len__(self):
        return len(self.flat)

    def __getitem__(self, idx):
        entry, crop_idx = self.flat[idx]
        path = entry['path']
        qtype, fname = path.split('/')

        img_path = os.path.join(self.crop_root, qtype, fname.replace('.png', f'_crop{crop_idx}.png'))

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        answers = entry.get('answer', [])
        if crop_idx < len(answers) and answers[crop_idx] and len(answers[crop_idx]) > 0:
            ans = answers[crop_idx][0]
        else:
            ans = ""  

        seq = torch.tensor(self.vocab.encode(ans), dtype=torch.long)

        return img, seq, len(seq), path, crop_idx



import matplotlib.pyplot as plt
import torchvision.transforms as T

def visualize_sample(dataset, index, vocab):
    img, seq, seq_len, path, crop_i = dataset[index]

    decoded_ans = vocab.decode(seq.tolist())

    plt.figure(figsize=(3, 3))
    img_show = T.ToPILImage()(img)  
    plt.imshow(img_show)
    plt.title(f'Answer: {decoded_ans}\nForm: {path}, Crop: {crop_i}')
    plt.axis('off')
    plt.show()



#  Model definitions
class Encoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.cnn = EfficientNet.from_pretrained('efficientnet-b0')
        in_f = self.cnn._fc.in_features
        self.cnn._fc = nn.Linear(in_f, embed_size)

    def forward(self, x):
        return self.cnn(x)

class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers=3, nhead=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, tgt, memory, tgt_mask):
        emb = self.embed(tgt)
        out = self.transformer(emb, memory, tgt_mask)
        return self.fc(out)

class CaptionModel(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super().__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, imgs, seqs, teacher_forcing_ratio=0.5):
        B, max_len = imgs.size(0), seqs.size(1)
        device = imgs.device
        feats = self.encoder(imgs)            
        memory = feats.unsqueeze(1)          
        outputs = torch.zeros(B, max_len, self.vocab_size, device=device)
        input_seq = seqs[:, :1]               
        for t in range(1, max_len):
            tgt_len = input_seq.size(1)
            mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
            out = self.decoder(input_seq, memory, tgt_mask=mask)
            outputs[:, t] = out[:, -1]
            if random.random() < teacher_forcing_ratio and t < max_len:
                input_seq = seqs[:, :t+1]
            else:
                next_token = out[:, -1].argmax(1, keepdim=True)
                input_seq = torch.cat([input_seq, next_token], dim=1)
        return outputs

def collate_fn(batch):
    imgs, seqs, lengths, paths, cis = zip(*batch)
    imgs = torch.stack(imgs)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    return imgs, seqs, lengths, paths, cis

def compute_cer(pred_str, true_str):
    if len(true_str) == 0:
        return float(len(pred_str) > 0)
    dist = editdistance.eval(pred_str, true_str)
    return dist / len(true_str)

def prepare_coco_refs(dataset, vocab):
    print("Preparing COCO references...")
    refs = {}
    for i in range(len(dataset)):
        img, seq, l, path, ci = dataset[i]
        key = f"{path}_crop{ci}"
        text = vocab.decode(seq.tolist())
        refs[key] = [text]
    return refs

def prepare_coco_res(model, dataset, vocab, device):
    print("Preparing COCO results...")
    res = {}
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    model.eval()
    with torch.no_grad():
        for imgs, seqs, lengths, paths, cis in tqdm(loader, desc="Generating predictions"):
            imgs = imgs.to(device)
            seqs = seqs.to(device)
            outputs = model(imgs, seqs, teacher_forcing_ratio=0)
            preds = outputs.argmax(2).cpu().numpy()
            for b in range(len(paths)):
                key = f"{paths[b]}_crop{cis[b]}"
                text = vocab.decode(preds[b])
                res[key] = [text]
    return res

#  Training and Evaluation functions
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, seqs, *_ in tqdm(loader, desc="Training"):
        imgs, seqs = imgs.to(device), seqs.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, seqs)
        loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), seqs[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device, vocab):
    model.eval()
    total_loss = 0
    all_true = []
    all_pred = []
    seq_exact = 0
    total_seq = 0
    total_cer = 0
    length_stats = {}
    qtype_stats = {}

    with torch.no_grad():
        for imgs, seqs, lengths, paths, cis in tqdm(loader, desc="Evaluating"):
            imgs, seqs = imgs.to(device), seqs.to(device)
            outputs = model(imgs, seqs, teacher_forcing_ratio=0)
            loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), seqs[:, 1:].reshape(-1))
            total_loss += loss.item()
            preds = outputs.argmax(2)
            for i in range(preds.size(0)):
                tgt_seq = seqs[i].cpu().tolist()
                pred_seq = preds[i].cpu().tolist()
                # Trim at EOS or PAD
                if vocab.token2idx['<EOS>'] in tgt_seq:
                    true_tokens = tgt_seq[1:tgt_seq.index(vocab.token2idx['<EOS>'])]
                else:
                    true_tokens = tgt_seq[1:]
                if vocab.token2idx['<EOS>'] in pred_seq:
                    pred_tokens = pred_seq[1:pred_seq.index(vocab.token2idx['<EOS>'])]
                else:
                    pred_tokens = pred_seq[1:len(true_tokens)+1]
                true_str = vocab.decode([vocab.token2idx['<SOS>']] + true_tokens + [vocab.token2idx['<EOS>']])
                pred_str = vocab.decode([vocab.token2idx['<SOS>']] + pred_tokens + [vocab.token2idx['<EOS>']])
                # Token-level
                min_len = min(len(true_tokens), len(pred_tokens))
                all_true.extend(true_tokens[:min_len])
                all_pred.extend(pred_tokens[:min_len])
                # Sequence exact
                if true_tokens == pred_tokens:
                    seq_exact += 1
                total_seq += 1
                # CER
                cer = compute_cer(pred_str, true_str)
                total_cer += cer
                # Length stats
                l = len(true_tokens)
                length_stats.setdefault(l, {'count':0, 'exact':0, 'cer_sum':0})
                length_stats[l]['count'] += 1
                if true_tokens == pred_tokens:
                    length_stats[l]['exact'] += 1
                length_stats[l]['cer_sum'] += cer
                # Qtype stats
                qtype = paths[i].split('/')[0]
                qtype_stats.setdefault(qtype, {'count':0, 'exact':0})
                qtype_stats[qtype]['count'] += 1
                if true_tokens == pred_tokens:
                    qtype_stats[qtype]['exact'] += 1
    token_acc = accuracy_score(all_true, all_pred) if all_true else 0
    token_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0) if all_true else 0
    seq_exact_rate = seq_exact / total_seq if total_seq>0 else 0
    avg_cer = total_cer / total_seq if total_seq>0 else 0
    length_summary = {
        l: {
            'count': info['count'],
            'exact_acc': info['exact']/info['count'] if info['count']>0 else 0,
            'avg_cer': info['cer_sum']/info['count'] if info['count']>0 else 0
        } for l, info in length_stats.items()
    }
    qtype_summary = {
        q: {
            'count': info['count'],
            'exact_acc': info['exact']/info['count'] if info['count']>0 else 0
        } for q, info in qtype_stats.items()
    }
    return total_loss/len(loader), token_acc, token_f1, seq_exact_rate, avg_cer, length_summary, qtype_summary

#  BLEU evaluation function
def compute_bleu_scores(refs, res):
    if COCO_EVAL_AVAILABLE:
        try:
            bleu_scorer = CocoBleu(4)
            score, _ = bleu_scorer.compute_score(refs, res)
            return {'BLEU-1': score[0], 'BLEU-2': score[1], 'BLEU-3': score[2], 'BLEU-4': score[3]}
        except Exception as e:
            print(f"COCO BLEU failed: {e}")
    if NLTK_AVAILABLE:
        list_refs = [[list(r[0])] for r in refs.values()]
        list_res = [list(res[k][0]) for k in res]
        weights = [(1,0,0,0),(0.5,0.5,0,0),(0.33,0.33,0.33,0),(0.25,0.25,0.25,0.25)]
        scores = {}
        for i, w in enumerate(weights, start=1):
            try:
                sc = corpus_bleu(list_refs, list_res, weights=w)
                scores[f'BLEU-{i}'] = sc
            except Exception as e:
                print(f"BLEU-{i} failed: {e}")
                scores[f'BLEU-{i}'] = 0
        return scores
    return {'BLEU': 'Not available'}

print("Utilities defined!\n")

print("Configuring experiment and preparing data...")
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
embed_size = 512
lr = 1e-4
epochs = 20

try:
    from google.colab import drive
    drive.mount('/content/drive')
    root = '/content/drive/MyDrive/Aligned_Sheets'
    print("Google Drive mounted!")
except ImportError:
    root = './Aligned_Sheets'
    print("Using local directory")

json_path = os.path.join(root, 'dataset_updated.json')
crop_dir = os.path.join(root, 'cropped_bboxes512_2_padded')

entries = load_entries(json_path)
print(f"Total entries: {len(entries)}")
vocab = Vocab()
for e in entries:
    for ans_list in e.get('answer', []):
        for ans in ans_list:
            vocab.add_sentence(ans)
print(f"Vocabulary size: {len(vocab)}")

#  SAVE VOCABULARY TO DISK 
import pickle

vocab_path = os.path.join(root, 'vocab.pkl')
os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)
print(f"[✔] Vocabulary saved to: {vocab_path}")

random.shuffle(entries)
N = len(entries)
train_e = entries[:int(0.8*N)]
val_e = entries[int(0.8*N):int(0.9*N)]
test_e = entries[int(0.9*N):]

print(f"Train entries: {len(train_e)}, Val entries: {len(val_e)}, Test entries: {len(test_e)}")

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_ds = BubbleDataset(train_e, vocab, crop_dir, tfm)
val_ds = BubbleDataset(val_e, vocab, crop_dir, tfm)
test_ds = BubbleDataset(test_e, vocab, crop_dir, tfm)


train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
print("Data preparation complete!\n")

# DEBUG CODE 
print("=== DEBUGGING DATASET ISSUE ===")

for idx, (entry, ci) in enumerate(test_ds.flat):
    if entry['path'] == "048/1234567892535.png" and ci == 3:
        print("FOUND at flat index =", idx)
        print(" entry path: ", entry['path'])
        print(" crop index: ", ci)
        print(" nforms:     ", entry.get('nforms'))
        print(" answer[{}]: ".format(ci), entry['answer'][ci] if ci < len(entry['answer']) else "INDEX OUT OF RANGE")
        break
else:
    print("Matching entry not found.")

target_entries = [e for e in test_e if e['path'] == "048/1234567892535.png"]
if target_entries:
    entry = target_entries[0]
    print("\n=== DETAILED DEBUG FOR 048/1234567892535.png ===")
    print("Path:", entry['path'])
    print("Nforms:", entry.get('nforms'))
    print("Answer array length:", len(entry.get('answer', [])))
    for i, ans_list in enumerate(entry.get('answer', [])):
        print(f"  answer[{i}]: {ans_list}")
    print("=" * 50)
else:
    print("Entry 048/1234567892535.png not found in test set")

print("=== END DEBUG ===\n")


visualize_sample(test_ds,10, vocab)



# DEBUG 
img, seq, length, path, ci = test_ds[291]
print("→ img path:", os.path.join(crop_dir, *path.split('/')[0:1], path.split('/')[1].replace('.png', f'_crop{ci}.png')))
print("→ returned path,ci:", path, ci)
print("→ returned GT answer:", vocab.decode(seq.tolist()))

print("Initializing model and starting training...")
model = CaptionModel(embed_size, len(vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx['<PAD>'])
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

train_losses, val_losses = [], []
val_token_accs, val_token_f1s, val_seq_exacts, val_avg_cers = [], [], [], []
val_length_summaries, val_qtype_summaries = [], []

best_f1 = 0
for ep in range(1, epochs + 1):
    print(f"\nEpoch {ep}/{epochs}")
    t_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    v_loss, v_acc, v_f1, v_seq_exact, v_avg_cer, length_summary, qtype_summary = eval_epoch(
        model, val_loader, criterion, device, vocab
    )
    train_losses.append(t_loss)
    val_losses.append(v_loss)
    val_token_accs.append(v_acc)
    val_token_f1s.append(v_f1)
    val_seq_exacts.append(v_seq_exact)
    val_avg_cers.append(v_avg_cer)
    val_length_summaries.append(length_summary)
    val_qtype_summaries.append(qtype_summary)
    scheduler.step()

    print(f"Epoch {ep} Results:")
    print(f"  Train Loss: {t_loss:.4f}")
    print(f"  Val Loss: {v_loss:.4f}")
    print(f"  Token Acc: {v_acc:.4f}")
    print(f"  Token F1: {v_f1:.4f}")
    print(f"  Seq Exact: {v_seq_exact:.4f}")
    print(f"  Avg CER: {v_avg_cer:.4f}")

    if v_f1 > best_f1:
        best_f1 = v_f1
        save_path = '/content/drive/MyDrive/Aligned_Sheets/imagecap_model_last.pth'
        torch.save(model.state_dict(), save_path)
        print(f"  New best F1: {best_f1:.4f} - Model saved at {save_path}")

print("\nTraining completed!")

print("\nLoading best model for evaluation...")
model.load_state_dict(torch.load('/content/drive/MyDrive/Aligned_Sheets/imagecap_model_last.pth'))

try:
    refs = prepare_coco_refs(val_ds, vocab)
    res = prepare_coco_res(model, val_ds, vocab, device)
    bleu_scores = compute_bleu_scores(refs, res)
    print("\nBLEU Scores:")
    for k, v in bleu_scores.items():
        print(f"{k}: {v:.4f}")
except Exception as e:
    print(f"Evaluation failed: {e}")

print("\nGenerating training plots...")
epochs_range = list(range(1, epochs+1))
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_losses, 'b-o', label='Train')
plt.plot(epochs_range, val_losses, 'r-o', label='Validation')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

metrics = [
    ('Token Accuracy', val_token_accs),
    ('Token F1', val_token_f1s),
    ('Sequence Exact Match', val_seq_exacts),
    ('Average CER', val_avg_cers)
]

plt.figure(figsize=(15, 10))
for i, (name, values) in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    plt.plot(epochs_range, values, 'g-o')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel(name.split()[-1])
    plt.grid(True)
plt.tight_layout()
plt.show()

if len(test_ds) > 0:
    print("\nVisualizing test examples...")
    samples = random.sample(range(len(test_ds)), min(5, len(test_ds)))
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(samples):
        img, seq, l, path, ci = test_ds[idx]
        img_vis = img.clone()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_vis = img_vis * std + mean
        img_vis = torch.clamp(img_vis, 0, 1)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img_vis.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"GT: {vocab.decode(seq.tolist())}\n{path}_crop{ci}")

        model.eval()
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device), seq.unsqueeze(0).to(device), teacher_forcing_ratio=0)
            pred = out.argmax(2).squeeze().cpu().numpy()
            text = vocab.decode(pred)
        plt.subplot(2, 5, i + 6)
        plt.imshow(img_vis.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Pred: {text}")
    plt.tight_layout()
    plt.show()

print("\nAll tasks completed!")

if len(test_ds) > 0:
    print("\nVisualizing test examples...")
    samples = random.sample(range(len(test_ds)), min(5, len(test_ds)))
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(samples):
        img, seq, l, path, ci = test_ds[idx]
        img_vis = img.clone()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_vis = img_vis * std + mean
        img_vis = torch.clamp(img_vis, 0, 1)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img_vis.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"GT: {vocab.decode(seq.tolist())}\n{path}_crop{ci}")

        model.eval()
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device), seq.unsqueeze(0).to(device), teacher_forcing_ratio=0)
            pred = out.argmax(2).squeeze().cpu().numpy()
            text = vocab.decode(pred)
        plt.subplot(2, 5, i + 6)
        plt.imshow(img_vis.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Pred: {text}")
    plt.tight_layout()
    plt.show()

print("\nAll tasks completed!")