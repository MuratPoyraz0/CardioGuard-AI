import io
import json
import os

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool

# ------------------------------------------------------------------
# 1. PAGE CONFIG + STYLE
# ------------------------------------------------------------------
st.set_page_config(
    page_title="CardioGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .hero {
        background: linear-gradient(135deg, #1f2a44 0%, #3b1f44 100%);
        padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2rem;}
    .hero p {margin: .4rem 0 0 0; opacity: .85;}
    .risk-card {
        padding: 1.2rem 1.4rem; border-radius: 14px; color: white; margin-bottom: .8rem;
    }
    .risk-high {background: #c0392b;}
    .risk-mid  {background: #d68910;}
    .risk-low  {background: #1e8449;}
    .risk-card h2 {margin: 0; font-size: 1.5rem;}
    .risk-card p {margin: .3rem 0 0 0; opacity: .95;}
    .disclaimer {
        border-left: 4px solid #d68910; background: rgba(214,137,16,.10);
        padding: .8rem 1rem; border-radius: 6px; font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 2. LITERATURE REFERENCES (labels only, NOT probabilities)
# ------------------------------------------------------------------
REFERENCE_LABELS = {
    "CC(=O)NC1=CC=C(C=C1)O": ("Paracetamol", "Non-blocker"),
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": ("Ibuprofen", "Non-blocker"),
    "CC(=O)OC1=CC=CC=C1C(=O)O": ("Aspirin", "Non-blocker"),
    "CN(C)C(=N)NC(=N)N": ("Metformin", "Non-blocker"),
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": ("Caffeine", "Non-blocker"),
    "CC(C)(C)C1=CC=C(C=C1)C(O)CCCN2CCC(CC2)C(O)(C3=CC=CC=C3)C4=CC=CC=C4": ("Terfenadine", "Known hERG blocker (withdrawn from market)"),
    "COC1=CC=C(C=C1)CCN2CCC(CC2)NC3=NC4=CC=CC=C4N3CC5=CC=C(C=C5)F": ("Astemizole", "Known hERG blocker (withdrawn from market)"),
    "OC1(CCN(CCCC(=O)C2=CC=C(F)C=C2)CC1)C3=CC=C(Cl)C=C3": ("Haloperidol", "Known hERG blocker"),
    "CS(=O)(=O)NC1=CC=C(C=C1)CCN(C)CCOC2=CC=C(C=C2)NS(C)(=O)=O": ("Dofetilide", "Known hERG blocker"),
    "CC(C)NCC(O)COC1=CC=C(CC(N)=O)C=C1": ("Atenolol", "Non-blocker"),
    "NC1=NC2=C(N=CN2COCCO)C(=O)N1": ("Acyclovir", "Non-blocker"),
}


def _inchikey(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol else None


REFERENCE_BY_KEY = {_inchikey(s): v for s, v in REFERENCE_LABELS.items() if _inchikey(s)}


def lookup_reference(mol):
    return REFERENCE_BY_KEY.get(Chem.MolToInchiKey(mol))


# ------------------------------------------------------------------
# 3. MODEL (Checkpoint boyutları olan [64 -> 32 -> 1] mimarisine ayarlandı)
# ------------------------------------------------------------------
class GIN(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=1):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv2 = GINConv(nn2)
        nn3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv3 = GINConv(nn3)
        
        # Checkpoint: fc1 weight [32, 64], fc2 weight [1, 32]
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@st.cache_resource
def load_model(path="gnn_herg_model.pt"):
    """Loads the model and reports whether the weights were really loaded."""
    model = GIN()
    if not os.path.exists(path):
        return None, f"File '{path}' not found.", [], []
    try:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
            
        model.load_state_dict(state, strict=True)
        model.eval()
        return model, None, [], []
    except Exception as e:
        return None, f"Could not load model: {e}", [], []


model, model_error, missing_keys, unexpected_keys = load_model()


@st.cache_data
def load_metrics(path="metrics.json"):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


metrics = load_metrics()


# ------------------------------------------------------------------
# 4. MOLECULE -> GRAPH
# ------------------------------------------------------------------
def smiles_to_graph(mol):
    feats = [
        [
            a.GetAtomicNum(),
            a.GetDegree(),
            a.GetFormalCharge(),
            int(a.GetIsAromatic()),
            a.GetTotalNumHs(),
        ]
        for a in mol.GetAtoms()
    ]
    x = torch.tensor(feats, dtype=torch.float)
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [(i, j), (j, i)]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


# ------------------------------------------------------------------
# 5. MODEL PREDICTION + REAL XAI (gradient saliency)
# ------------------------------------------------------------------
def model_predict(mol, with_saliency=False):
    data = smiles_to_graph(mol)
    x = data.x.clone().requires_grad_(with_saliency)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    out = model(x, data.edge_index, batch)
    prob = float(torch.sigmoid(out)[0, 0].detach())
    saliency = None
    if with_saliency:
        model.zero_grad()
        out[0, 0].backward()
        if x.grad is not None:
            saliency = (x.grad * x).sum(dim=1).detach().numpy()
    return prob, saliency


# ------------------------------------------------------------------
# 6. RULE-BASED PHARMACOPHORE SCORE
# ------------------------------------------------------------------
BASIC_AMINE = Chem.MolFromSmarts(
    "[NX3;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-C#N);!$(N-N);!$(N-O)]"
)
ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")


def rule_score(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    arom = Lipinski.NumAromaticRings(mol)
    n_basic = len(mol.GetSubstructMatches(BASIC_AMINE))
    has_acid = mol.HasSubstructMatch(ACID)

    score, notes = 0.15, []

    if n_basic >= 1:
        score += 0.25
        notes.append(("Basic amine present", "+0.25"))
    if arom >= 2:
        score += 0.20
        notes.append((f"{arom} aromatic rings", "+0.20"))
    elif arom == 1:
        score += 0.08
        notes.append(("1 aromatic ring", "+0.08"))
    if logp > 3.0:
        score += 0.15
        notes.append((f"High lipophilicity (logP {logp:.1f})", "+0.15"))
    elif logp < 1.0:
        score -= 0.10
        notes.append((f"Low lipophilicity (logP {logp:.1f})", "-0.10"))
    if mw > 350:
        score += 0.05
        notes.append((f"Large molecule (MW {mw:.0f})", "+0.05"))
    if tpsa < 60:
        score += 0.10
        notes.append((f"Low polarity (TPSA {tpsa:.0f})", "+0.10"))
    elif tpsa > 90:
        score -= 0.10
        notes.append((f"High polarity (TPSA {tpsa:.0f})", "-0.10"))
    if has_acid:
        score -= 0.25
        notes.append(("Carboxylic acid group", "-0.25"))

    desc = dict(mw=mw, logp=logp, tpsa=tpsa, arom=arom, n_basic=n_basic, acid=has_acid)
    return float(min(max(score, 0.02), 0.98)), notes, desc


# ------------------------------------------------------------------
# 7. APPLICABILITY DOMAIN
# ------------------------------------------------------------------
ALLOWED = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def domain_warnings(mol):
    w = []
    mw = Descriptors.MolWt(mol)
    if mw < 150 or mw > 700:
        w.append(f"Molecular weight ({mw:.0f}) is outside the typical drug-like range (150-700).")
    if any(a.GetAtomicNum() not in ALLOWED for a in mol.GetAtoms()):
        w.append("Contains unusual elements (metals etc.), prediction is unreliable.")
    if len(Chem.GetMolFrags(mol)) > 1:
        w.append("Contains multiple separate fragments (may be a salt or mixture).")
    return w


# ------------------------------------------------------------------
# 8. COMBINING
# ------------------------------------------------------------------
def analyze(smiles, model_weight, with_saliency=False):
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    r_score, r_notes, desc = rule_score(mol)
    if model is not None:
        m_prob, sal = model_predict(mol, with_saliency)
        w = model_weight
    else:
        m_prob, sal, w = None, None, 0.0
    final = r_score if m_prob is None else w * m_prob + (1 - w) * r_score
    gap = None if m_prob is None else abs(m_prob - r_score)
    return dict(
        mol=mol, model=m_prob, rule=r_score, notes=r_notes, desc=desc,
        final=final, gap=gap, saliency=sal,
        warnings=domain_warnings(mol), ref=lookup_reference(mol),
    )


def risk_level(p):
    if p >= 0.60:
        return "risk-high", "🚨 HIGH RISK", "High likelihood of being an hERG blocker"
    if p >= 0.40:
        return "risk-mid", "⚠️ UNCERTAIN / MODERATE RISK", "No clear verdict, experimental testing is a priority"
    return "risk-low", "✅ LOW RISK", "Low likelihood of being an hERG blocker"


def confidence_text(gap):
    if gap is None:
        return "Low", "Only the rule engine ran (model failed to load)."
    if gap < 0.20:
        return "High", "Model and rule engine agree."
    if gap < 0.40:
        return "Medium", "Model and rule engine partly disagree."
    return "Low", "Model and rule engine conflict, treat the result with caution."


def saliency_image(mol, sal):
    max_abs = float(max(abs(sal).max(), 1e-9))
    weights = [float(s / max_abs) for s in sal]
    drawer = Draw.MolDraw2DCairo(560, 360)
    SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, draw2d=drawer)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


# ------------------------------------------------------------------
# 9. SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🫀 CardioGuard AI")
    st.caption("Early-stage hERG cardiotoxicity screening tool")
    st.divider()

    st.markdown("### 🧠 Model")
    if model is None:
        st.error(model_error or "Model could not be loaded.")
        if missing_keys:
            st.caption(f"Missing keys: {missing_keys}")
    else:
        st.success("Model weights loaded")
        st.caption("Architecture: 3-layer GIN · 5 atom features · TDC hERG data")

    st.markdown("### 📊 Validation")
    if metrics:
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        c2.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        st.caption(metrics.get("note", "Test set metrics"))
    else:
        st.info(
            "No measured test metrics yet. After training, create a `metrics.json` "
            "and the real values will appear here."
        )

    st.divider()
    st.markdown("### ⚙️ Advanced")
    model_weight = st.slider(
        "Model weight", 0.0, 1.0, 0.5, 0.05,
        help="Final score = w × GNN + (1 − w) × rule engine. "
             "Increase it once you have measured and trust the model.",
        disabled=model is None,
    )

    st.divider()
    st.markdown("### 🧪 Example molecules")
    examples = {
        "Terfenadine": "CC(C)(C)C1=CC=C(C=C1)C(O)CCCN2CCC(CC2)C(O)(C3=CC=CC=C3)C4=CC=CC=C4",
        "Astemizole": "COC1=CC=C(C=C1)CCN2CCC(CC2)NC3=NC4=CC=CC=C4N3CC5=CC=C(C=C5)F",
        "Haloperidol": "OC1(CCN(CCCC(=O)C2=CC=C(F)C=C2)CC1)C3=CC=C(Cl)C=C3",
        "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
        "Naproxen": "COC1=CC2=CC(C(C)C(=O)O)=CC=C2C=C1",
        "Amoxicillin": "CC1(C)SC2C(NC(=O)C(N)C3=CC=C(O)C=C3)C(=O)N2C1C(=O)O",
    }
    for name, smi in examples.items():
        if st.button(name, use_container_width=True, key=f"ex_{name}"):
            st.session_state["smiles"] = smi

# ------------------------------------------------------------------
# 10. MAIN PAGE
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🫀 CardioGuard AI</h1>
      <p>A decision-support tool that estimates the hERG potassium channel blockade risk
      of molecules that have not yet been synthesized, at the earliest stage of drug discovery.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer">
    <b>Important:</b> This is a <b>screening and prioritization</b> tool. It does not replace
    experimental testing (in vitro hERG assays, clinical QT studies) and must not be used for
    clinical decisions. Predictions cover hERG blockade only, not other cardiac risk mechanisms.
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

tab_single, tab_batch, tab_about = st.tabs(["🔬 Single molecule", "📋 Batch analysis", "ℹ️ How it works"])

# ---------------- SINGLE MOLECULE ----------------
with tab_single:
    smiles_input = st.text_input(
        "SMILES structure",
        key="smiles",
        placeholder="e.g. CC(=O)NC1=CC=C(C=C1)O",
    )
    run = st.button("🚀 Analyze", type="primary")

    if run:
        if not smiles_input.strip():
            st.error("Please enter a SMILES string.")
        else:
            res = analyze(smiles_input, model_weight, with_saliency=True)
            if res is None:
                st.error("Invalid SMILES. Please check the input.")
            else:
                css, title, sub = risk_level(res["final"])
                conf, conf_note = confidence_text(res["gap"])

                left, right = st.columns([1, 1.3])
                with left:
                    st.markdown(
                        f'<div class="risk-card {css}"><h2>{title}</h2>'
                        f'<p>{sub}</p><p><b>Estimated blocker score: {res["final"]*100:.1f}%</b></p></div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(res["final"], 0.0), 1.0))
                    st.markdown(f"**Confidence:** {conf} — {conf_note}")

                    m1, m2 = st.columns(2)
                    m1.metric("GNN model", "—" if res["model"] is None else f'{res["model"]:.2f}')
                    m2.metric("Rule engine", f'{res["rule"]:.2f}')

                    if res["ref"]:
                        st.info(f"📚 **Literature reference:** {res['ref'][0]} — {res['ref'][1]}")
                    for w in res["warnings"]:
                        st.warning(w)

                with right:
                    if res["saliency"] is not None:
                        st.markdown("#### 🔍 Atom contribution map (gradient saliency)")
                        st.image(saliency_image(res["mol"], res["saliency"]))
                        st.caption(
                            "This map is computed from the GNN's own gradients. "
                            "Darker regions are the atoms that influence the model output most. "
                            "It shows what the model looks at, not proof of causality."
                        )
                    else:
                        st.warning("The atom map could not be generated because the model failed to load.")

                with st.expander("🧮 How was the rule-engine score calculated?", expanded=True):
                    st.table(
                        pd.DataFrame(
                            [("Baseline", "0.15")] + res["notes"], columns=["Factor", "Contribution"]
                        )
                    )
                    d = res["desc"]
                    st.caption(
                        f"MW {d['mw']:.0f} · logP {d['logp']:.2f} · TPSA {d['tpsa']:.0f} · "
                        f"aromatic rings {d['arom']} · basic amines {d['n_basic']}"
                    )

# ---------------- BATCH ANALYSIS ----------------
with tab_batch:
    st.markdown("One SMILES per line (optional format: `name,SMILES`).")
    batch_text = st.text_area("Molecules", height=180, placeholder="Aspirin,CC(=O)OC1=CC=CC=C1C(=O)O")
    if st.button("📋 Run batch analysis"):
        rows = []
        for line in batch_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if "," in line:
                name, smi = line.split(",", 1)
            else:
                name, smi = line[:20], line
            res = analyze(smi, model_weight)
            if res is None:
                rows.append({"Name": name.strip(), "Final score": None, "Level": "Invalid SMILES"})
                continue
            _, title, _ = risk_level(res["final"])
            rows.append({
                "Name": name.strip(),
                "Final score": round(res["final"], 3),
                "GNN": None if res["model"] is None else round(res["model"], 3),
                "Rule": round(res["rule"], 3),
                "Level": title,
                "Confidence": confidence_text(res["gap"])[0],
                "Literature": res["ref"][1] if res["ref"] else "—",
            })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "cardioguard_results.csv")

# ---------------- ABOUT ----------------
with tab_about:
    st.markdown(
        """
### How does the system work?
1. **GNN (GIN):** The molecule is modeled as a graph (atoms = nodes, bonds = edges) and the model outputs a blocker probability.
2. **Rule engine:** Looks at known hERG pharmacophore features (basic amine, aromatic rings, lipophilicity, polarity, acid group). Every point is shown transparently.
3. **Combination:** The final score is a weighted average of the two components. You can change the weight in the sidebar.
4. **Confidence:** If the model and the rule engine agree, confidence is high. If they conflict, the tool says so openly.
5. **XAI:** The atom map is computed from the model's gradients.

### Limitations
- hERG is not the only cardiac risk mechanism (Nav1.5, Cav1.2, metabolites, dose etc. are out of scope).
- Reliability drops on chemical scaffolds that differ strongly from the training data.
- The rule engine is heuristic, it is not something learned during model training.
- Results require experimental validation.
"""
    )
