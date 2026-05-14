import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import timm
import os

st.set_page_config(
    page_title="The Truth Herald — Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=IM+Fell+English:ital@0;1&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Code+Pro:wght@400;600&display=swap');

/* ─── GLOBAL ─── */
html, body, .stApp {
    background: #f2ede4 !important;
    color: #1a1008 !important;
}
.block-container {
    max-width: 760px !important;
    padding: 0 1.5rem 4rem !important;
}
* { box-sizing: border-box; }

/* ─── HIDE STREAMLIT CHROME ─── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ─── MASTHEAD ─── */
.masthead {
    border-top: 6px double #1a1008;
    border-bottom: 1px solid #1a1008;
    padding: 0.4rem 0;
    margin-bottom: 0.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.mast-meta {
    font-family: 'Source Code Pro', monospace;
    font-size: 9px;
    letter-spacing: 1.5px;
    color: #5a4a3a;
    text-transform: uppercase;
}
.mast-edition {
    font-family: 'Source Code Pro', monospace;
    font-size: 9px;
    letter-spacing: 1px;
    color: #5a4a3a;
    text-align: right;
}

/* ─── NAMEPLATE ─── */
.nameplate {
    text-align: center;
    border-bottom: 6px double #1a1008;
    padding: 0.6rem 0 0.8rem;
    margin-bottom: 0.3rem;
}
.paper-name {
    font-family: 'Playfair Display', serif;
    font-size: 52px;
    font-weight: 900;
    color: #1a1008;
    letter-spacing: -1px;
    line-height: 1;
}
.paper-motto {
    font-family: 'IM Fell English', serif;
    font-style: italic;
    font-size: 13px;
    color: #5a4a3a;
    margin-top: 4px;
    letter-spacing: 0.5px;
}

/* ─── SECTION RULE ─── */
.section-rule {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1rem 0;
}
.section-rule::before, .section-rule::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a1008;
}
.section-rule span {
    font-family: 'Source Code Pro', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #5a4a3a;
    white-space: nowrap;
}

/* ─── COLUMN LAYOUT ─── */
.col-wrapper {
    display: grid;
    grid-template-columns: 1fr 3px 2fr;
    gap: 0 1.2rem;
    margin: 1rem 0;
}
.col-divider {
    background: #1a1008;
    width: 1px;
    margin: 0 auto;
}

/* ─── HEADLINE ─── */
.story-hed {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 900;
    color: #1a1008;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.story-dek {
    font-family: 'Libre Baskerville', serif;
    font-style: italic;
    font-size: 13px;
    color: #3a2a1a;
    line-height: 1.5;
    margin-bottom: 1rem;
    border-left: 3px solid #c0392b;
    padding-left: 10px;
}
.byline {
    font-family: 'Source Code Pro', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    color: #5a4a3a;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #c0c0b0;
}

/* ─── SIDEBAR INFO ─── */
.info-box-paper {
    background: #e8e0d0;
    border: 1px solid #b0a898;
    padding: 0.8rem;
    margin-bottom: 0.8rem;
}
.info-box-label {
    font-family: 'Source Code Pro', monospace;
    font-size: 8px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5a4a3a;
    border-bottom: 1px solid #b0a898;
    padding-bottom: 4px;
    margin-bottom: 6px;
}
.info-box-body {
    font-family: 'Libre Baskerville', serif;
    font-size: 11px;
    color: #2a1a0a;
    line-height: 1.6;
}
.accuracy-badge {
    display: inline-block;
    background: #c0392b;
    color: #f2ede4;
    font-family: 'Source Code Pro', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    margin-top: 6px;
    letter-spacing: 1px;
}

/* ─── MODE SELECTOR ─── */
.mode-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    border: 1px solid #1a1008;
    margin-bottom: 1rem;
}
.mode-option {
    padding: 0.7rem 0.5rem;
    text-align: center;
    border-right: 1px solid #1a1008;
    cursor: pointer;
}
.mode-option:last-child { border-right: none; }
.mode-option.active { background: #1a1008; color: #f2ede4; }
.mode-name-text {
    font-family: 'Source Code Pro', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.mode-acc-text {
    font-family: 'Libre Baskerville', serif;
    font-style: italic;
    font-size: 10px;
    color: #c0392b;
    margin-top: 2px;
}
.mode-option.active .mode-acc-text { color: #f0a090; }

/* ─── INPUT FIELDS ─── */
.stTextArea textarea {
    background: #ede6d8 !important;
    border: 1px solid #8a7a6a !important;
    border-radius: 0 !important;
    font-family: 'Libre Baskerville', serif !important;
    font-size: 13px !important;
    color: #1a1008 !important;
    padding: 0.8rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea::placeholder { color: #8a7a6a !important; font-style: italic; }
.stTextArea textarea:focus {
    border-color: #1a1008 !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextArea label {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #5a4a3a !important;
    font-weight: 400 !important;
}

/* ─── FILE UPLOADER ─── */
.stFileUploader {
    background: #ede6d8 !important;
    border: 1px dashed #8a7a6a !important;
    border-radius: 0 !important;
    padding: 0.5rem !important;
}
.stFileUploader label {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #5a4a3a !important;
}

/* ─── BUTTON ─── */
.stButton button {
    background: #1a1008 !important;
    color: #f2ede4 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton button:hover {
    background: #c0392b !important;
    color: #f2ede4 !important;
}

/* ─── RADIO ─── */
.stRadio label {
    font-family: 'Libre Baskerville', serif !important;
    font-size: 13px !important;
    color: #1a1008 !important;
}
.stRadio > div { gap: 0.5rem !important; }

/* ─── VERDICT BANNER ─── */
.verdict-fake {
    background: #1a1008;
    color: #f2ede4;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
    border-left: 8px solid #c0392b;
}
.verdict-real {
    background: #1a3018;
    color: #d4ead4;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
    border-left: 8px solid #27ae60;
}
.verdict-stamp {
    font-family: 'Playfair Display', serif;
    font-size: 36px;
    font-weight: 900;
    line-height: 1;
    flex-shrink: 0;
}
.verdict-detail-text { flex: 1; }
.verdict-conf-num {
    font-family: 'Source Code Pro', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    opacity: 0.7;
    margin-bottom: 2px;
}
.verdict-conf-val {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
}

/* ─── PROBABILITY BARS ─── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
}
.prob-label {
    font-family: 'Source Code Pro', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5a4a3a;
    width: 40px;
    flex-shrink: 0;
}
.prob-track {
    flex: 1;
    background: #d8d0c0;
    height: 8px;
}
.prob-fill-fake { background: #c0392b; height: 100%; }
.prob-fill-real { background: #27ae60; height: 100%; }
.prob-pct {
    font-family: 'Source Code Pro', monospace;
    font-size: 10px;
    font-weight: 600;
    color: #1a1008;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

/* ─── FOOTER RULE ─── */
.footer-rule {
    border: none;
    border-top: 6px double #1a1008;
    margin: 2rem 0 0.5rem;
}
.footer-text {
    font-family: 'Source Code Pro', monospace;
    font-size: 9px;
    letter-spacing: 1.5px;
    color: #5a4a3a;
    text-align: center;
    text-transform: uppercase;
}

/* ─── SPINNER ─── */
.stSpinner > div { border-top-color: #c0392b !important; }

/* ─── SUCCESS/WARNING ─── */
.stAlert {
    border-radius: 0 !important;
    font-family: 'Libre Baskerville', serif !important;
    font-size: 13px !important;
}

/* ─── DIVIDER ─── */
hr { border: none; border-top: 1px solid #b0a898; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──
MODELS_DIR       = "models"
TEXT_FUSION_PATH = os.path.join(MODELS_DIR, "fusion-best (3).pt")
IMG_FUSION_PATH  = os.path.join(MODELS_DIR, "fusion_image-best.pt")
MULTIMODAL_PATH  = os.path.join(MODELS_DIR, "final_multimodal_v7-best (3).pt")
TOKENIZER_PATH   = os.path.join(MODELS_DIR, "hybrid_tokenizer (3).pkl")

ROBERTA_NAME = "roberta-base"
MAX_LEN      = 128
LABEL2ID     = {"FALSE": 0, "TRUE": 1}
ID2LABEL     = {0: "FALSE", 1: "TRUE"}
TEXT_DIM     = 896
IMAGE_DIM    = 2304
COMBINED_DIM = TEXT_DIM + IMAGE_DIM
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE CLASSES (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, lstm_outputs):
        scores  = self.v(torch.tanh(self.attn(lstm_outputs))).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        return (lstm_outputs * weights.unsqueeze(-1)).sum(dim=1), weights

class HybridBiLSTM_Attn_GRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=256,
                 hidden_dim=128, num_gru_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm     = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                                  bidirectional=True, dropout=dropout)
        self.attn       = AttentionLayer(hidden_dim)
        self.gru        = nn.GRU(hidden_dim*2, hidden_dim,
                                 num_layers=num_gru_layers,
                                 batch_first=True, bidirectional=False,
                                 dropout=dropout)
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, 128),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2)
        )
    def encode(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.bilstm(x)
        attn_vec, _ = self.attn(lstm_out)
        gru_out, _  = self.gru(attn_vec.unsqueeze(1))
        return gru_out[:, -1, :]
    def forward(self, x):
        return self.fc(self.encode(x))

class FusionTextModel(nn.Module):
    def __init__(self, roberta_model, hybrid_model, dropout=0.3):
        super().__init__()
        self.roberta    = roberta_model
        self.hybrid     = hybrid_model
        roberta_dim     = self.roberta.config.hidden_size
        hybrid_dim      = self.hybrid.hidden_dim
        fusion_dim      = roberta_dim + hybrid_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, 2)
        )
    def forward(self, input_ids, attention_mask, hyb_seq):
        with torch.no_grad():
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            cls_vec = outputs.hidden_states[-1][:, 0, :]
        with torch.no_grad():
            gru_vec = self.hybrid.encode(hyb_seq)
        fused  = torch.cat([cls_vec, gru_vec], dim=1)
        return self.classifier(fused)

class EfficientNetB3Classifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        base             = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone    = base.features
        self.pool        = base.avgpool
        self.feature_dim = base.classifier[1].in_features
        self.classifier  = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512), nn.SiLU(inplace=True),
            nn.Dropout(p=dropout/2), nn.Linear(512, num_classes)
        )
    def encode(self, x):
        return torch.flatten(self.pool(self.backbone(x)), 1)
    def forward(self, x):
        return self.classifier(self.encode(x))

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone    = timm.create_model("vit_base_patch16_384", pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.classifier  = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512), nn.GELU(),
            nn.Dropout(p=dropout/2), nn.Linear(512, num_classes)
        )
    def encode(self, x):
        return self.backbone(x)
    def forward(self, x):
        return self.classifier(self.encode(x))

class FusionImageModel(nn.Module):
    def __init__(self, effnet, vit, num_classes=2, dropout=0.4):
        super().__init__()
        self.effnet      = effnet
        self.vit         = vit
        self.feature_dim = 2304
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 1024), nn.GELU(),
            nn.Dropout(p=dropout/2), nn.Linear(1024, num_classes)
        )
    def encode(self, img_eff, img_vit):
        feat_eff = self.effnet.encode(img_eff)
        feat_vit = self.vit.encode(img_vit)
        return torch.cat([feat_eff, feat_vit], dim=1)
    def forward(self, img_eff, img_vit):
        return self.fusion_head(self.encode(img_eff, img_vit))

class FusionTextModelV7(nn.Module):
    def __init__(self, roberta_model, hybrid_model, dropout=0.3):
        super().__init__()
        self.roberta    = roberta_model
        self.hybrid     = hybrid_model
        roberta_dim     = self.roberta.config.hidden_size
        hybrid_dim      = self.hybrid.hidden_dim
        fusion_dim      = roberta_dim + hybrid_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, 2)
        )
    def encode(self, input_ids, attention_mask, hyb_seq):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_vec = outputs.hidden_states[-1][:, 0, :]
        gru_vec = self.hybrid.encode(hyb_seq)
        return torch.cat([cls_vec, gru_vec], dim=1)

class FinalMultimodalModelV7(nn.Module):
    def __init__(self, text_fusion, image_fusion, dropout=0.3):
        super().__init__()
        self.text_fusion  = text_fusion
        self.image_fusion = image_fusion
        self.classifier = nn.Sequential(
            nn.Linear(COMBINED_DIM, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 2)
        )
    def forward(self, input_ids, attention_mask, hyb_seq, img_eff, img_vit):
        text_feat = self.text_fusion.encode(input_ids, attention_mask, hyb_seq)
        with torch.no_grad():
            img_feat = self.image_fusion.encode(img_eff, img_vit)
        fused = torch.cat([text_feat, img_feat], dim=1)
        return self.classifier(fused)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_text_fusion_model():
    with open(TOKENIZER_PATH, "rb") as f:
        keras_tokenizer = pickle.load(f)
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_NAME)
    state = torch.load(TEXT_FUSION_PATH, map_location="cpu", weights_only=True)
    vocab_size = state["hybrid.embedding.weight"].shape[0]
    hidden_dim = state["hybrid.gru.weight_ih_l0"].shape[1] // 2
    roberta_base = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
    hybrid_base = HybridBiLSTM_Attn_GRU(
        vocab_size=vocab_size, embed_dim=256, hidden_dim=hidden_dim, num_gru_layers=1, dropout=0.3)
    model = FusionTextModel(roberta_base, hybrid_base)
    model.load_state_dict(state)
    model.eval()
    return model, roberta_tokenizer, keras_tokenizer

@st.cache_resource
def load_image_fusion_model():
    ckpt        = torch.load(IMG_FUSION_PATH, map_location="cpu", weights_only=True)
    effnet_base = EfficientNetB3Classifier(num_classes=2, dropout=0.3)
    vit_base    = ViTClassifier(num_classes=2, dropout=0.3)
    model       = FusionImageModel(effnet_base, vit_base)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

@st.cache_resource
def load_multimodal_model():
    with open(TOKENIZER_PATH, "rb") as f:
        keras_tokenizer = pickle.load(f)
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_NAME)
    ckpt  = torch.load(MULTIMODAL_PATH, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]
    vocab_size = state["text_fusion.hybrid.embedding.weight"].shape[0]
    hidden_dim = state["text_fusion.hybrid.gru.weight_ih_l0"].shape[1] // 2
    roberta_base = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
    hybrid_base  = HybridBiLSTM_Attn_GRU(
        vocab_size=vocab_size, embed_dim=256, hidden_dim=hidden_dim, num_gru_layers=1, dropout=0.3)
    text_fusion  = FusionTextModelV7(roberta_base, hybrid_base)
    effnet_base  = EfficientNetB3Classifier(num_classes=2, dropout=0.3)
    vit_base     = ViTClassifier(num_classes=2, dropout=0.3)
    image_fusion = FusionImageModel(effnet_base, vit_base)
    model        = FinalMultimodalModelV7(text_fusion, image_fusion, dropout=0.2)
    model.load_state_dict(state)
    model.eval()
    return model, roberta_tokenizer, keras_tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_text(text, roberta_tok, keras_tok):
    from keras.preprocessing.sequence import pad_sequences
    rob_enc = roberta_tok(
        text, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
    seq     = keras_tok.texts_to_sequences([text])
    hyb_arr = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    hyb_seq = torch.tensor(hyb_arr, dtype=torch.long)
    return rob_enc["input_ids"], rob_enc["attention_mask"], hyb_seq

def preprocess_image(image):
    eff_tf = T.Compose([T.Resize((380, 380)), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    vit_tf = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return eff_tf(image).unsqueeze(0), vit_tf(image).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def show_result(pred, conf, prob_fake, prob_real, mode_name, accuracy):
    if pred == 0:
        st.markdown(f"""
        <div class="verdict-fake">
            <div class="verdict-stamp">Fake</div>
            <div class="verdict-detail-text">
                <div class="verdict-conf-num">CONFIDENCE SCORE</div>
                <div class="verdict-conf-val">{conf:.1f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-real">
            <div class="verdict-stamp">Real</div>
            <div class="verdict-detail-text">
                <div class="verdict-conf-num">CONFIDENCE SCORE</div>
                <div class="verdict-conf-val">{conf:.1f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="prob-row">
        <div class="prob-label">Fake</div>
        <div class="prob-track"><div class="prob-fill-fake" style="width:{prob_fake:.1f}%"></div></div>
        <div class="prob-pct">{prob_fake:.1f}%</div>
    </div>
    <div class="prob-row">
        <div class="prob-label">Real</div>
        <div class="prob-track"><div class="prob-fill-real" style="width:{prob_real:.1f}%"></div></div>
        <div class="prob-pct">{prob_real:.1f}%</div>
    </div>
    <div class="info-box-paper" style="margin-top:1rem">
        <div class="info-box-label">Analysis Report</div>
        <div class="info-box-body">
            Model: {mode_name}<br>
            Reported Accuracy: <span style="font-weight:700;color:#c0392b">{accuracy}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import datetime
    today = datetime.date.today().strftime("%A, %B %d, %Y").upper()

    st.markdown(f"""
    <div class="masthead">
        <div class="mast-meta">Vol. I &nbsp;·&nbsp; Est. 2026 &nbsp;·&nbsp; IFND Dataset</div>
        <div class="mast-edition">{today}<br>FAKE NEWS DETECTOR</div>
    </div>
    <div class="nameplate">
        <div class="paper-name">The Truth Herald</div>
        <div class="paper-motto">"All the News That's Fit to Verify"</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-rule"><span>Detection Mode</span></div>', unsafe_allow_html=True)

    mode = st.radio(
        label="Select Mode",
        options=["📝 Text Only", "🖼️ Image Only", "🔗 Text + Image (Best Accuracy)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-rule"><span>Analysis Desk</span></div>', unsafe_allow_html=True)

    # ── MODE 1: TEXT ONLY ──
    if mode == "📝 Text Only":
        col_info, _, col_main = st.columns([1, 0.05, 2])
        with col_info:
            st.markdown("""
            <div class="info-box-paper">
                <div class="info-box-label">Text Model</div>
                <div class="info-box-body">RoBERTa + Hybrid BiLSTM fusion architecture.</div>
                <div class="accuracy-badge">97.29% ACC</div>
            </div>
            <div class="info-box-paper">
                <div class="info-box-label">Tip</div>
                <div class="info-box-body" style="font-style:italic">Paste the full article body for best results.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_main:
            with st.spinner("Loading Text Model..."):
                text_model, roberta_tok, keras_tok = load_text_fusion_model()
            st.success("Text Model Ready")
            news_text = st.text_area("PASTE ARTICLE TEXT", placeholder="Enter news article here...", height=160)
            if st.button("ANALYSE - SUBMIT TO EDITORIAL"):
                if not news_text.strip():
                    st.warning("Please enter some text.")
                    return
                with st.spinner("Composing verdict..."):
                    input_ids, attention_mask, hyb_seq = preprocess_text(news_text, roberta_tok, keras_tok)
                    with torch.no_grad():
                        logits = text_model(input_ids, attention_mask, hyb_seq)
                        probs  = torch.softmax(logits, dim=1)[0]
                        pred   = torch.argmax(probs).item()
                        conf   = probs[pred].item() * 100
                show_result(pred, conf, probs[0].item()*100, probs[1].item()*100,
                            "Text Fusion (RoBERTa + Hybrid)", "97.29%")

    # ── MODE 2: IMAGE ONLY ──
    elif mode == "🖼️ Image Only":
        col_info, _, col_main = st.columns([1, 0.05, 2])
        with col_info:
            st.markdown("""
            <div class="info-box-paper">
                <div class="info-box-label">Image Model</div>
                <div class="info-box-body">EfficientNetB3 + ViT dual-encoder fusion.</div>
                <div class="accuracy-badge">76.17% ACC</div>
            </div>
            <div class="info-box-paper">
                <div class="info-box-label">Tip</div>
                <div class="info-box-body" style="font-style:italic">JPG, JPEG or PNG images accepted.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_main:
            with st.spinner("Loading Image Model..."):
                image_model = load_image_fusion_model()
            st.success("Image Model Ready")
            uploaded_image = st.file_uploader("UPLOAD NEWS IMAGE", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                st.image(uploaded_image, caption="Submitted for review", use_container_width=True)
            if st.button("ANALYSE — SUBMIT TO EDITORIAL"):
                if uploaded_image is None:
                    st.warning("Please upload an image.")
                    return
                with st.spinner("Composing verdict..."):
                    image = Image.open(uploaded_image).convert("RGB")
                    img_eff, img_vit = preprocess_image(image)
                    with torch.no_grad():
                        logits = image_model(img_eff, img_vit)
                        probs  = torch.softmax(logits, dim=1)[0]
                        pred   = torch.argmax(probs).item()
                        conf   = probs[pred].item() * 100
                show_result(pred, conf, probs[0].item()*100, probs[1].item()*100,
                            "Image Fusion (EfficientNetB3 + ViT)", "76.17%")

    # ── MODE 3: TEXT + IMAGE ──
    else:
        col_info, _, col_main = st.columns([1, 0.05, 2])
        with col_info:
            st.markdown("""
            <div class="info-box-paper">
                <div class="info-box-label">Multimodal</div>
                <div class="info-box-body">V7 - Simple Concat + Partial Unfreeze strategy.</div>
                <div class="accuracy-badge">97.39% ACC ★</div>
            </div>
            <div class="info-box-paper">
                <div class="info-box-label">Best Mode</div>
                <div class="info-box-body" style="font-style:italic">Combines both text and image signals for maximum accuracy.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_main:
            with st.spinner("Loading Multimodal Model..."):
                mm_model, roberta_tok, keras_tok = load_multimodal_model()
            st.success("Multimodal Model Ready")
            news_text = st.text_area("PASTE ARTICLE TEXT", placeholder="Enter news article here...", height=120)
            uploaded_image = st.file_uploader("UPLOAD NEWS IMAGE", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                st.image(uploaded_image, caption="Submitted for review", use_container_width=True)
            if st.button("ANALYSE - SUBMIT TO EDITORIAL"):
                if not news_text.strip():
                    st.warning("Please enter some text.")
                    return
                if uploaded_image is None:
                    st.warning("Please upload an image.")
                    return
                with st.spinner("Composing verdict..."):
                    input_ids, attention_mask, hyb_seq = preprocess_text(news_text, roberta_tok, keras_tok)
                    image   = Image.open(uploaded_image).convert("RGB")
                    img_eff, img_vit = preprocess_image(image)
                    with torch.no_grad():
                        logits = mm_model(input_ids, attention_mask, hyb_seq, img_eff, img_vit)
                        probs  = torch.softmax(logits, dim=1)[0]
                        pred   = torch.argmax(probs).item()
                        conf   = probs[pred].item() * 100
                show_result(pred, conf, probs[0].item()*100, probs[1].item()*100,
                            "Multimodal V7 - Simple Concat + Partial Unfreeze", "97.39%")

    st.markdown("""
    <hr class="footer-rule">
    <div class="footer-text">Built with PyTorch &amp; Streamlit &nbsp;·&nbsp; IFND Dataset &nbsp;·&nbsp; The Truth Herald @2026</div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()