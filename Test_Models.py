import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import torchvision.models as models 
import tkinter as tk
from tkinter import filedialog, messagebox
import pickle

# BƯỚC 1: CẤU HÌNH THIẾT BỊ VÀ TỪ ĐIỂN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy trên thiết bị: {device}")

try:
    with open('vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
        vocab = vocab_data['vocab']
        inv_vocab = vocab_data['inv_vocab']
        print("Đã tải từ điển thành công!")
except FileNotFoundError:
    print("⚠️ WARNING: The vocab.pkl file was not found. Please ensure you have saved the dictionary from your Notebook file!")
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    inv_vocab = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

# BƯỚC 2: Các models được định nghĩa trong Notebook đã được copy vào đây để đảm bảo tính nhất quán khi tải trọng số .pth vào các mô hình này.
class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(300, 512, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return h[-1]

#CNN + LSTM Train from scratch
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.fc = nn.Linear(256 * 14 * 14, 512)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class VQAModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.cnn = CNNEncoder()
        self.q_enc = QuestionEncoder(vocab_size)

        self.fusion = nn.Linear(1024, 512)

        self.embedding = nn.Embedding(vocab_size, 300)
        self.decoder = nn.LSTM(300, 512, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, image, question, answer):
        img_feat = self.cnn(image)
        q_feat = self.q_enc(question)

        fused = torch.cat([img_feat, q_feat], dim=1)
        fused = self.fusion(fused)

        emb = self.embedding(answer)

        h0 = fused.unsqueeze(0)
        c0 = torch.zeros_like(h0)

        out, _ = self.decoder(emb, (h0, c0))
        return self.fc(out)

# CNN + LSTM + Attention
class CNNEncoder_Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.size()

        x = x.view(B, C, H*W)
        x = x.permute(0, 2, 1)

        return x
    
class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()

        self.attn = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, features, hidden):

        B, N, F = features.size()

        hidden = hidden.unsqueeze(1).expand(-1, N, -1)

        energy = torch.tanh(
            self.attn(torch.cat((features, hidden), dim=2))
        )

        attention = self.v(energy).squeeze(2)

        alpha = torch.softmax(attention, dim=1)

        context = torch.sum(features * alpha.unsqueeze(2), dim=1)

        return context, alpha

class VQAModel_Attention(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.cnn = CNNEncoder_Attention()
        self.q_enc = QuestionEncoder(vocab_size)

        self.attention = Attention(256, 512)

        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(300 + 256, 512, batch_first=True)

        self.fc = nn.Linear(512, vocab_size)

    def forward(self, img, ques, ans):
        features = self.cnn(img)
        q_feat = self.q_enc(ques)

        hidden = q_feat.unsqueeze(0)
        cell = torch.zeros_like(hidden)

        embeddings = self.embedding(ans)

        outputs = []

        for t in range(embeddings.size(1)):
            hidden_state = hidden[-1]
            context, _ = self.attention(features, hidden_state)

            lstm_input = torch.cat((embeddings[:, t, :], context), dim=1)
            lstm_input = lstm_input.unsqueeze(1)

            out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

            out = self.fc(out.squeeze(1))
            outputs.append(out)

        return torch.stack(outputs, dim=1)

# ResNet50 + LSTM
class CNNEncoder_Pretrained(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for param in resnet.parameters():
            param.requires_grad = False


        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)

        features = self.dropout(features)
        features = self.fc(features)
        return features
class VQAModel_Pretrained(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.cnn = CNNEncoder_Pretrained(embed_size=512)
        self.q_enc = QuestionEncoder(vocab_size)

        self.embedding = nn.Embedding(vocab_size, 300)

        self.lstm = nn.LSTM(300 + 512, 512, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, img, ques, ans):
        img_feat = self.cnn(img)
        q_feat = self.q_enc(ques)

        hidden = q_feat.unsqueeze(0)
        cell = torch.zeros_like(hidden)

        embeddings = self.embedding(ans)

        img_feat_expanded = img_feat.unsqueeze(1).expand(-1, embeddings.size(1), -1)

        lstm_input = torch.cat((embeddings, img_feat_expanded), dim=2)

        out, _ = self.lstm(lstm_input, (hidden, cell))
        out = self.fc(out)

        return out
    
# ResNet50 + LSTM + Attention
class ResNet50_Attention_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.reduce_dim = nn.Conv2d(2048, 256, kernel_size=1)

    def forward(self, images):
        x = self.resnet(images)
        x = self.reduce_dim(x)

        B, C, H, W = x.size()
        x = x.view(B, C, H * W)
        x = x.permute(0, 2, 1)

        return x
class VQAModel_Pretrained_Attention(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.cnn = ResNet50_Attention_Encoder()
        self.q_enc = QuestionEncoder(vocab_size)

        self.attention = Attention(256, 512)

        self.embedding = nn.Embedding(vocab_size, 300)

        self.lstm = nn.LSTM(300 + 256, 512, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, img, ques, ans):
        features = self.cnn(img)
        q_feat = self.q_enc(ques)

        hidden = q_feat.unsqueeze(0)
        cell = torch.zeros_like(hidden)

        embeddings = self.embedding(ans)

        outputs = []

        for t in range(embeddings.size(1)):
            hidden_state = hidden[-1]

            context, _ = self.attention(features, hidden_state)

            lstm_input = torch.cat((embeddings[:, t, :], context), dim=1)
            lstm_input = lstm_input.unsqueeze(1)

            out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            out = self.fc(out.squeeze(1))

            outputs.append(out)

        return torch.stack(outputs, dim=1)

# BƯỚC 3: KHỞI TẠO VÀ LOAD TRỌNG SỐ 4 MÔ HÌNH
print("Loading model payloads into RAM/VRAM...")
vocab_size = len(vocab) 

model_1 = VQAModel(vocab_size).to(device)
model_2 = VQAModel_Attention(vocab_size).to(device)
model_3 = VQAModel_Pretrained(vocab_size).to(device)
model_4 = VQAModel_Pretrained_Attention(vocab_size).to(device)

try:
    model_1.load_state_dict(torch.load("best_model_base.pth", map_location=device))
    model_2.load_state_dict(torch.load("best_model_attention.pth", map_location=device))
    model_3.load_state_dict(torch.load("best_model_pretrained.pth", map_location=device))
    model_4.load_state_dict(torch.load("best_model_pretrained_attention.pth", map_location=device))
    
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    print("Model downloaded successfully! Ready to predict.")
except Exception as e:
    print(f"⚠️ Unable to load .pth file (Please check Steps 2 and 3 again): {e}")

# BƯỚC 4: HÀM TIỀN XỬ LÝ VÀ DỰ ĐOÁN (INFERENCE)
MAX_LEN = 20

def tokenize(text):
    return text.lower().split()

def encode_question(text):
    tokens = tokenize(text)
    seq = [vocab.get(w, vocab["<UNK>"]) for w in tokens]
    seq = [vocab["<SOS>"]] + seq + [vocab["<EOS>"]]

    if len(seq) < MAX_LEN:
        seq += [vocab["<PAD>"]] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]

    return torch.tensor(seq)

def decode_sequence(seq, inv_vocab):
    words = []
    for idx in seq:
        word = inv_vocab.get(idx.item(), "")
        if word == "<EOS>":
            break
        if word not in ["<PAD>", "<SOS>"]:
            words.append(word)
    return " ".join(words)

def generate_answer(model, img, ques, vocab, inv_vocab, device, max_len=20):
    model.eval()
    img = img.unsqueeze(0).to(device)
    ques = ques.unsqueeze(0).to(device)
    input_seq = torch.tensor([[vocab["<SOS>"]]]).to(device)
    outputs = []

    for _ in range(max_len):
        with torch.no_grad():
            out = model(img, ques, input_seq)
        
        next_token = out[:, -1, :].argmax(-1)
        token = next_token.item()

        if token == vocab["<EOS>"]:
            break

        outputs.append(token)
        input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)

    return decode_sequence(torch.tensor(outputs), inv_vocab)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_answer(model, image_path, question_text):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img)
        ques_tensor = encode_question(question_text)
        answer = generate_answer(model, img_tensor, ques_tensor, vocab, inv_vocab, device)
        return answer if answer.strip() else "(Model can not generate an answer)"
    except Exception as e:
        return f"ERROE: {str(e)}"

# BƯỚC 5: XÂY DỰNG GIAO DIỆN TKINTER
window = tk.Tk()
window.title("Hệ thống VQA - Midterm for Deep Learning")
window.geometry("850x750")

current_image_path = None 

# --- GIAO DIỆN ---
tk.Label(window, text="Please select a photo.:", font=("Helvetica", 14, "bold"), fg='blue').pack(pady=5)
btn_select = tk.Button(window, text="📁 Select an image from your computer", command=lambda: select_image(), font=("Helvetica", 10))
btn_select.pack()

tk.Label(window, text="Please enter your question:", font=("Helvetica", 14, "bold"), fg='blue').pack(pady=5)
txt_question = tk.Entry(window, font=("Helvetica", 14), width=50)
txt_question.pack()

btn_compare = tk.Button(window, text="🚀 Compare 4 Models", command=lambda: run_models(), font=("Helvetica", 12, "bold"), bg="orange")
btn_compare.pack(pady=10)

img_canvas = tk.Canvas(window, width=300, height=300, bg="lightgray", highlightthickness=1, highlightbackground="black")
img_canvas.pack(pady=10)

tk.Label(window, text="Predicted results:", font=("Helvetica", 14, "bold"), fg='blue').pack(pady=5)
lbl_res_1 = tk.Label(window, text="Model 1 (Base): Waiting...", font=("Helvetica", 12))
lbl_res_1.pack(pady=2)
lbl_res_2 = tk.Label(window, text="Model 2 (Attention): Waiting...", font=("Helvetica", 12))
lbl_res_2.pack(pady=2)
lbl_res_3 = tk.Label(window, text="Model 3 (ResNet50): Waiting...", font=("Helvetica", 12))
lbl_res_3.pack(pady=2)
lbl_res_4 = tk.Label(window, text="Model 4 (ResNet50 + Att): Waiting...", font=("Helvetica", 13, "bold"), fg="green")
lbl_res_4.pack(pady=2)

# --- CÁC HÀM XỬ LÝ SỰ KIỆN ---
def select_image():
    global current_image_path
    filepath = filedialog.askopenfilename(
        title="Select an image for prediction",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
    )
    if filepath:
        current_image_path = filepath
        img_canvas.delete("all")
        img = Image.open(filepath)
        img.thumbnail((300, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        img_canvas.create_image(150, 150, anchor=tk.CENTER, image=img_tk)
        img_canvas.image = img_tk 
        window.update()

def run_models():
    global current_image_path
    if not current_image_path:
        messagebox.showwarning("Warning", "You forgot to select an image!")
        return
    
    question = txt_question.get().strip()
    if not question:
        messagebox.showwarning("Warning", "You haven't entered a question!")
        return
    
    lbl_res_1.config(text="Model 1 (Base): Thinking...", fg="black")
    lbl_res_2.config(text="Model 2 (Attention): Thinking...", fg="black")
    lbl_res_3.config(text="Model 3 (ResNet50): Thinking...", fg="black")
    lbl_res_4.config(text="Model 4 (ResNet50 + Att): Thinking...", fg="black")
    window.update() 

    ans_1 = predict_answer(model_1, current_image_path, question)
    ans_2 = predict_answer(model_2, current_image_path, question)
    ans_3 = predict_answer(model_3, current_image_path, question)
    ans_4 = predict_answer(model_4, current_image_path, question)
    

    lbl_res_1.config(text=f"Model 1 (Base): {ans_1}")
    lbl_res_2.config(text=f"Model 2 (Attention): {ans_2}")
    lbl_res_3.config(text=f"Model 3 (ResNet50): {ans_3}")
    lbl_res_4.config(text=f"Model 4 (ResNet50 + Att): {ans_4}", fg="green")

window.mainloop()