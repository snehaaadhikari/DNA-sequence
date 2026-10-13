"""Streamlit app: is this DNA human or chimpanzee?

Same logic as identify() in your notebook: find the closest gene, require >= 97% identity to a reference,
then decide using the k-mers that differ between the human and chimp copies of that gene.

    streamlit run app.py
"""
import html as htmllib
import os
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Human or Chimp DNA?", page_icon="🧬", layout="wide")

HUMAN, CHIMP = "Homo sapiens", "Pan troglodytes"
K_REF, SAMPLE_MOD = 15, 16
ENC = {"A": 0, "C": 1, "G": 2, "T": 3}
MASK = (1 << (2 * K_REF)) - 1

# --------------------------------------------------------------------------- look & feel
import base64

BASE_COL = {"A": "#C46F5C", "C": "#92CC9D", "G": "#7783B5", "T": "#DADDA8"}  # same palette as EDA


def helix_svg(width=1400, height=160, amp=48, period=260, strand_op=1.0, rung_op=0.9):
    """Double-helix drawing: two sine strands joined by colour-coded base-pair rungs."""
    import math
    mid, parts = height / 2, []
    pairs = [("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")]
    n = 0
    for x in range(10, width, 22):
        y = amp * math.sin(2 * math.pi * x / period)
        a, b = pairs[(n * 7 + n // 3) % 4]
        n += 1
        parts.append(f'<line x1="{x}" y1="{mid - y:.1f}" x2="{x}" y2="{mid + y:.1f}" stroke="{BASE_COL[a]}" '
                     f'stroke-width="4" stroke-linecap="round" opacity="{rung_op}"/>')
        parts.append(f'<circle cx="{x}" cy="{mid - y:.1f}" r="5" fill="{BASE_COL[a]}"/>'
                     f'<circle cx="{x}" cy="{mid + y:.1f}" r="5" fill="{BASE_COL[b]}"/>')
    for sign, col in ((1, "#2A9D8F"), (-1, "#264653")):
        pts = " ".join(f"{x},{mid + sign * amp * math.sin(2 * math.pi * x / period):.1f}" for x in range(0, width + 4, 4))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="5" '
                     f'stroke-linecap="round" opacity="{strand_op}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">{"".join(parts)}</svg>')


def svg_uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


HERO = svg_uri(helix_svg())
SIDE = svg_uri(helix_svg(width=900, height=120, amp=34, period=200, strand_op=0.35, rung_op=0.35))

CSS = r"""
:root { --ink:#16323b; --muted:#4f6b75; --teal:#2A9D8F; --navy:#264653; --line:#cfe5e1; --soft:#f4fbf9; }
/* force a light, readable look whatever the browser/Streamlit theme is */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#ffffff !important; color:var(--ink) !important; }
[data-testid="stHeader"] { background:rgba(255,255,255,.95) !important; }
[data-testid="stToolbar"] *, [data-testid="stStatusWidget"] * { color:var(--ink) !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
.stApp [data-testid="stMarkdownContainer"] *, .stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stWidgetLabel"] *, .stApp [data-testid="stRadio"] * { color:var(--ink); }
.stApp [data-testid="stCaptionContainer"] * { color:var(--muted); }

/* sidebar */
[data-testid="stSidebar"] { background:linear-gradient(180deg,#eef9f6 0%,#ffffff 100%) !important; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink) !important; }
[data-testid="stSidebar"]::after { content:""; display:block; height:120px; margin-top:20px;
    background:url("__SIDE__") center / 260% auto no-repeat; }

/* inputs */
textarea, input { background:#fbfefd !important; color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; }
[data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="base-input"] {
    background:#fbfefd !important; border:1px solid var(--line) !important; border-radius:10px !important; }
textarea { font-family:ui-monospace,Consolas,monospace !important; letter-spacing:1px; }
textarea::placeholder, input::placeholder { color:#8aa3ab !important; -webkit-text-fill-color:#8aa3ab !important; }

/* tabs */
button[data-baseweb="tab"] p { color:var(--muted) !important; font-weight:600; }
button[data-baseweb="tab"][aria-selected="true"] p { color:var(--teal) !important; }
[data-baseweb="tab-highlight"] { background:var(--teal) !important; }
[data-baseweb="tab-border"] { background:var(--line) !important; }

/* buttons, alerts, spinner, metrics */
div.stButton > button[kind="primary"] { background:linear-gradient(90deg,#2A9D8F,#264653) !important; border:0 !important;
    border-radius:10px; padding:.55rem 1.7rem; }
div.stButton > button[kind="primary"] * { color:#ffffff !important; }
[data-testid="stAlert"] { background:var(--soft) !important; border:1px solid var(--line); border-left:5px solid var(--teal); border-radius:10px; }
[data-testid="stAlert"] * { color:var(--ink) !important; }
[data-testid="stSpinner"] { background:var(--soft) !important; border:1px solid var(--line); border-radius:10px; padding:.5rem .9rem; }
[data-testid="stSpinner"] * { color:var(--ink) !important; }
[data-testid="stMetric"] { background:var(--soft); border:1px solid var(--line); border-radius:12px; padding:.7rem .95rem; }
[data-testid="stMetric"] * { color:var(--ink) !important; }
[data-testid="stTable"] *, [data-testid="stDataFrame"] { color:var(--ink) !important; }

/* custom tables, meters, charts */
table.vt { width:100%; border-collapse:separate; border-spacing:0; border:1px solid var(--line); border-radius:14px;
    overflow:hidden; background:#fff; font-size:.95rem; margin:6px 0 14px; }
table.vt th { background:#e9f7f4; color:var(--navy) !important; text-align:left; padding:.7rem .95rem; font-weight:700;
    border-bottom:1px solid var(--line); }
table.vt td { padding:.7rem .95rem; border-bottom:1px solid #edf5f3; color:var(--ink) !important; vertical-align:top; }
table.vt tbody tr:last-child td { border-bottom:0; }
table.vt tbody tr:nth-child(even) td { background:#fafdfc; }
.badge { display:inline-block; padding:3px 12px; border-radius:20px; font-weight:700; font-size:.85rem; white-space:nowrap; }
.b-unk { background:#fff1d6; color:#8a5a00 !important; border:1px solid #f2d28a; }
.b-forced { background:#eef1f3; color:#7b8d95 !important; text-decoration:line-through; }
.b-sp { background:#d8f0ec; color:#14665c !important; }
.b-other { background:#eef1f3; color:var(--ink) !important; }
.meter { margin:8px 0 4px; }
.mrow { display:flex; align-items:center; gap:14px; margin:16px 0; }
.mlab { width:215px; font-weight:600; color:var(--navy) !important; }
.mtrack { position:relative; flex:1; height:32px; background:#edf4f3; border-radius:9px; }
.mfill { position:absolute; left:0; top:0; bottom:0; border-radius:9px 0 0 9px; }
.mgap { position:absolute; top:0; bottom:0; display:flex; align-items:center; justify-content:center; box-sizing:border-box;
    background:repeating-linear-gradient(135deg,#fbe3e0,#fbe3e0 6px,#ffffff 6px,#ffffff 12px); border:1px dashed #e08a7e; }
.mgap span { font-size:.78rem; font-weight:800; color:#a3321f !important; white-space:nowrap; }
.mthr { position:absolute; top:-7px; bottom:-7px; width:3px; background:var(--navy); border-radius:2px; }
.mval { width:78px; text-align:right; font-weight:800; color:var(--navy) !important; }
.mleg { color:var(--muted) !important; font-size:.85rem; margin-top:2px; }
.sw { display:inline-block; width:16px; height:12px; vertical-align:middle; margin-right:5px; }
.sw.gap { background:repeating-linear-gradient(135deg,#fbe3e0,#fbe3e0 3px,#fff 3px,#fff 6px); border:1px dashed #e08a7e; }
.sw.thr { background:var(--navy); width:4px; }
.callout { background:var(--soft); border:1px solid var(--line); border-left:5px solid var(--teal); border-radius:10px;
    padding:.85rem 1.05rem; margin:12px 0 8px; color:var(--ink) !important; }
.bal { display:flex; flex:1; height:32px; border-radius:9px; overflow:hidden; }
.bal div { display:flex; align-items:center; padding:0 12px; white-space:nowrap; font-size:.82rem; font-weight:700; color:#fff !important; }
.grp { margin:6px 0 14px; }
.gl { font-weight:800; color:var(--navy) !important; font-size:1.05rem; margin-bottom:4px; }
.gbar { display:flex; align-items:center; gap:8px; margin:4px 0; }
.gn { width:48px; font-size:.82rem; color:var(--muted) !important; }
.gt { flex:1; background:#edf4f3; border-radius:6px; height:12px; }
.gf { height:12px; border-radius:6px; }
.gv { width:52px; text-align:right; font-size:.8rem; font-weight:700; color:var(--navy) !important; }

/* result card */
.result { display:flex; align-items:center; gap:24px; border-radius:20px; padding:24px 30px; margin:16px 0 12px; border:2px solid; }
.result .ico { font-size:3.2rem; line-height:1; }
.result .big { font-size:2.7rem; font-weight:900; letter-spacing:1px; line-height:1.05; }
.result .small { font-size:1rem; margin-top:2px; opacity:.9; }
.result .why { font-size:1.08rem; margin-top:10px; color:var(--ink) !important; }
.r-human { background:#e6f6f2; border-color:#8fd0c3; } .r-human .big { color:#14665c !important; }
.r-chimp { background:#fdeee9; border-color:#efb3a1; } .r-chimp .big { color:#a33f22 !important; }
.r-unk { background:#fff6e0; border-color:#efd08a; } .r-unk .big { color:#8a5a00 !important; }
.r-tell { background:#eef2f4; border-color:#c9d6dc; } .r-tell .big { color:#3f5c67 !important; }
.tiles { display:flex; gap:12px; flex-wrap:wrap; margin:4px 0 22px; }
.tile { flex:1; min-width:150px; background:var(--soft); border:1px solid var(--line); border-radius:12px; padding:.7rem 1rem; }
.tile .k { color:var(--muted) !important; font-size:.76rem; text-transform:uppercase; letter-spacing:.6px; }
.tile .v { color:var(--navy) !important; font-size:1.35rem; font-weight:800; }
/* hero */
.hero { border-radius:18px; padding:24px 30px 10px; margin-bottom:20px; border:1px solid var(--line);
    background:linear-gradient(135deg,#e9f7f4 0%,#ffffff 60%,#f4f6ff 100%); box-shadow:0 6px 24px rgba(38,70,83,.08); }
.hero h1 { margin:0; color:var(--navy) !important; font-size:2.2rem; letter-spacing:-.5px; }
.hero p { margin:4px 0 0; color:var(--muted) !important; }
.hero .helix { height:100px; margin-top:6px; background:url("__HERO__") center / cover no-repeat; }
.legend span { display:inline-block; padding:2px 12px; margin:0 6px 0 0; border-radius:20px; font-weight:700;
    color:var(--navy) !important; font-size:.85rem; }
"""
st.markdown("<style>" + CSS.replace("__HERO__", HERO).replace("__SIDE__", SIDE) + "</style>",
            unsafe_allow_html=True)


def hero(title, subtitle):
    legend = "".join(f'<span style="background:{c}55">{b}</span>' for b, c in BASE_COL.items())
    st.markdown(f'<div class="hero"><h1>🧬 {title}</h1><p>{subtitle}</p>'
                f'<div class="helix"></div><div class="legend">{legend}</div><br></div>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- sidebar
DATA_DIR = st.sidebar.text_input("Data folder", "DataSets")
MIN_ID = st.sidebar.slider("Min. match to call a species", 0.80, 1.00, 0.97, 0.01,
                           help="Same 97% rule as identify() in your notebook.")

# --------------------------------------------------------------------------- reference method (from your notebook)
def kmers(seq, k=K_REF):
    out, code, valid = set(), 0, 0
    for ch in seq:
        v = ENC.get(ch)
        if v is None:
            code, valid = 0, 0
            continue
        code = ((code << 2) | v) & MASK
        valid += 1
        if valid >= k:
            out.add(code)
    return out


def build_references(df):
    df = df[df["class"].isin([HUMAN, CHIMP])].copy()
    df["gene_key"] = df["gene"].astype(str).str.upper()
    per_gene = {}
    for gene, cls, seq in zip(df["gene_key"], df["class"], df["sequence"]):
        d = per_gene.setdefault(gene, {HUMAN: set(), CHIMP: set()})
        d[cls] |= kmers(str(seq).upper())
    diag, anchor, names = [], {}, []
    for gene, d in per_gene.items():
        h, c = d[HUMAN], d[CHIMP]
        if not h or not c:
            continue
        gid = len(names)
        names.append(gene)
        diag.append((h - c, c - h))
        for code in (h | c):
            if code % SAMPLE_MOD == 0:
                anchor[code] = gid if anchor.get(code, gid) == gid else -1
    return diag, anchor, names


def classify_ref(seq, diag, anchor):
    q = kmers(seq)
    votes = Counter(anchor[c] for c in q if c % SAMPLE_MOD == 0 and anchor.get(c, -1) >= 0)
    if not votes:
        return None, None, 0, 0
    gid = votes.most_common(1)[0][0]
    h_only, c_only = diag[gid]
    nh, nc = len(q & h_only), len(q & c_only)
    if nh == nc:
        return None, gid, nh, nc
    return (HUMAN if nh > nc else CHIMP), gid, nh, nc


@st.cache_resource(show_spinner="Building reference k-mer index (one-time, may take a few minutes)...")
def load_refs(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "all_combined_dna.csv"),
                     usecols=["sequence", "class", "gene"]).dropna(subset=["sequence"])
    df["sequence"] = df["sequence"].str.upper()
    return build_references(df)




@st.cache_resource(show_spinner="Loading reference genes (one-time)...")
def load_gene_seqs(data_dir):
    """gene -> {species: [all transcript sequences]} (identity is checked against every transcript)."""
    df = pd.read_csv(os.path.join(data_dir, "all_combined_dna.csv"),
                     usecols=["sequence", "class", "gene"]).dropna(subset=["sequence"])
    df = df[df["class"].isin([HUMAN, CHIMP])]
    out = {}
    for g, c, s in zip(df["gene"].astype(str).str.upper(), df["class"], df["sequence"].str.upper()):
        out.setdefault(g, {HUMAN: [], CHIMP: []})[c].append(s)
    return out


def _identity(q, ref):
    sm = SequenceMatcher(None, ref, q, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(q)


@st.cache_data(show_spinner="Checking against the reference genes...")
def identify(seq, data_dir, min_id):
    """Returns dict(status=HUMAN|CHIMP|UNKNOWN|CANNOT TELL, gene, top, nh, nc, reason)."""
    if len(seq) < 50:
        return dict(status="UNKNOWN", gene=None, top=None, reason="the sequence is shorter than 50 letters")
    diag, anchor, names = load_refs(data_dir)
    pred, gid, nh, nc = classify_ref(seq, diag, anchor)
    if gid is None:
        return dict(status="UNKNOWN", gene=None, top=None,
                    reason="no gene in your human/chimp data matches this sequence")
    gene = names[gid]
    best = {HUMAN: 0.0, CHIMP: 0.0}
    for cls, refs in load_gene_seqs(data_dir).get(gene, {}).items():
        for r in refs:
            best[cls] = max(best[cls], _identity(seq, r))
            if best[cls] >= 0.999:
                break
    top = max(best.values())
    if top < min_id:
        return dict(status="UNKNOWN", gene=gene, top=top, nh=nh, nc=nc,
                    reason=f"the closest gene ({gene}) is only {top:.1%} identical and at least {min_id:.0%} is needed")
    if pred is None:
        return dict(status="CANNOT TELL", gene=gene, top=top, nh=nh, nc=nc, reason="")
    return dict(status="HUMAN" if pred == HUMAN else "CHIMP", gene=gene, top=top, nh=nh, nc=nc, reason="")


def parse_fasta(text):
    recs, name, buf = [], "sequence", []
    for line in text.strip().splitlines():
        if line.startswith(">"):
            if buf:
                recs.append((name, "".join(buf)))
            name, buf = line[1:].strip() or f"seq{len(recs) + 1}", []
        else:
            buf.append("".join(line.split()).upper())
    if buf:
        recs.append((name, "".join(buf)))
    return recs


# --------------------------------------------------------------------------- page
def card(res):
    s, e = res["status"], htmllib.escape
    if s in ("HUMAN", "CHIMP"):
        mine, other = (res["nh"], res["nc"]) if s == "HUMAN" else (res["nc"], res["nh"])
        kind = "human" if s == "HUMAN" else "chimp"
        ico, small = ("🧑", "Homo sapiens") if s == "HUMAN" else ("🐒", "Pan troglodytes")
        why = (f"It matches the {kind} copy of the <b>{e(res['gene'])}</b> gene ({res['top']:.1%} identical) and carries "
               f"{mine} {kind}-specific DNA markers against {other} for the other species.")
        cls = f"r-{kind}"
    elif s == "CANNOT TELL":
        ico, small, cls = "⚖️", "Human and chimp are identical here", "r-tell"
        why = (f"It matches the <b>{e(res['gene'])}</b> gene ({res['top']:.1%} identical), but this stretch is the same "
               "in both species, so it can't be told apart. Try a longer sequence.")
    else:
        ico, small, cls = "❓", "Not recognised as human or chimpanzee", "r-unk"
        why = f"{e(res['reason']).capitalize()}. It may come from another species or a gene that isn't in your data."
    st.markdown(f'<div class="result {cls}"><div class="ico">{ico}</div><div><div class="big">{s}</div>'
                f'<div class="small">{small}</div><div class="why">{why}</div></div></div>', unsafe_allow_html=True)


def tiles(seq, res):
    gene = res.get("gene") or "none found"
    match = f"{res['top']:.1%}" if res.get("top") is not None else "n/a"
    items = [("Length", f"{len(seq):,} letters"), ("Closest gene", gene), ("Match to that gene", match)]
    st.markdown('<div class="tiles">' + "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{htmllib.escape(v)}</div></div>'
                                                for k, v in items) + "</div>", unsafe_allow_html=True)


hero("Is this DNA human or chimpanzee?", "Paste a sequence and get one clear answer.")
text = st.text_area("DNA sequence", height=170, placeholder=">my_sequence\nATGGCGTACGTTAGC...")

if st.button("Identify", type="primary"):
    records = parse_fasta(text) if text.strip() else []
    if not records:
        st.error("Paste a DNA sequence first.")
    for name, seq in records:
        if len(records) > 1:
            st.subheader(name)
        bad = sum(c not in "ACGTN" for c in seq)
        if bad > 0.05 * max(len(seq), 1):
            st.warning("This doesn't look like DNA: more than 5% of the letters are not A, C, G, T or N.")
        if len(seq) > 8000:
            st.caption("Long sequence: this can take a minute.")
        try:
            res = identify(seq, DATA_DIR, MIN_ID)
        except FileNotFoundError as e:
            st.error(f"Missing file: {e.filename}. Check the data folder in the sidebar.")
            break
        card(res)
        tiles(seq, res)

with st.expander("How is this decided?"):
    st.markdown(
        "1. The sequence is matched to the closest gene in your human and chimpanzee data.\n"
        f"2. If it is less than {MIN_ID:.0%} identical to that gene, it is called **UNKNOWN**: it is not human or chimp.\n"
        "3. Otherwise the app looks at the short DNA stretches (markers) that differ between the human and chimp copies "
        "of that gene and counts which species the sequence shares.")