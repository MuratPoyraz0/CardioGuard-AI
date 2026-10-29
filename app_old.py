import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
import numpy as np
from PIL import Image
import io

# Sayfa Yapılandırması
st.set_page_config(page_title="CardioGuard AI - hERG Toxicity Predictor", layout="wide")
st.title("🫀 CardioGuard AI: GNN Tabanlı hERG Toksisite & XAI Analizi")
st.markdown("Moleküler SMILES yapısını girerek hERG potasyum kanalı blokaj riskini ve atom-düzeyinde toksisite haritasını inceleyin.")

# 1. GIN Model Mimarisi
class GINNet(nn.Module):
    def __init__(self, in_features=5, hidden_dim=64, num_classes=1):
        super(GINNet, self).__init__()
        nn1 = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        nn3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv3 = GINConv(nn3)

        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# 2. Yardımcı Dönüştürücüler
def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs()
    ]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    x = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data, mol

# Modeli Yükleme
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GINNet(in_features=5, hidden_dim=64).to(device)
    try:
        model.load_state_dict(torch.load("gnn_herg_model.pt", map_location=device))
        model.eval()
    except Exception as e:
        st.warning("Model dosyası (gnn_herg_model.pt) yüklenemedi. Lütfen önce modeli kaydedin.")
    return model, device

model, device = load_model()

# Arayüz Girdisi
smiles_input = st.text_input("Molekül SMILES Kodunu Girin:", "")

if st.button("Toksisite Analizi Yap"):
    data, mol = smiles_to_graph(smiles_input)
    if data is None:
        st.error("Geçersiz SMILES girdisi! Lütfen geçerli bir kimyasal yapı girin.")
    else:
        # Tahmin Hesabı
        data = data.to(device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
        
        data.x.requires_grad = True
        out = model(data)
        prob = torch.sigmoid(out).item()
        
        # Gradyan Tabanlı XAI
        prob_tensor = torch.sigmoid(out)
        prob_tensor.backward()
        node_grads = data.x.grad.norm(dim=-1).cpu().numpy()

        # Sonuç Paneli
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tahmin Sonucu")
            risk_color = "red" if prob > 0.5 else "green"
            risk_label = "TOKSİK (hERG Bloker)" if prob > 0.5 else "GÜVENLİ (Non-Bloker)"
            st.markdown(f"### Durum: :{risk_color}[{risk_label}]")
            st.metric("hERG Blokaj Olasılığı", f"%{prob * 100:.2f}")

        with col2:
            st.subheader("XAI: Atom Katkı Önem Haritası")
            d2d = Draw.MolDraw2DCairo(400, 400)
            SimilarityMaps.GetSimilarityMapFromWeights(
                mol, 
                [float(w) for w in node_grads], 
                draw2d=d2d
            )
            d2d.FinishDrawing()
            img = Image.open(io.BytesIO(d2d.GetDrawingText()))
            st.image(img, caption="Kontur yoğunluğu yüksek olan bölgeler hERG blokaj riskini en çok artıran atom gruplarıdır.")
