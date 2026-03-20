import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import ndcg_score

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASCAR Race Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=Barlow:wght@300;400;500&display=swap');

/* ---------- global reset ---------- */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
}
.stApp {
    background: #0a0a0f;
    color: #e8e6df;
}
.block-container {
    padding: 2rem 2.5rem 4rem;
    max-width: 1400px;
}

/* ---------- header ---------- */
.nascar-header {
    display: flex;
    align-items: flex-end;
    gap: 1.2rem;
    padding-bottom: 1.4rem;
    border-bottom: 3px solid #f5c518;
    margin-bottom: 2rem;
}
.nascar-header h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    color: #f5f0e6;
    line-height: 1;
    margin: 0;
    text-transform: uppercase;
}
.nascar-header .accent {
    color: #f5c518;
}
.nascar-header .sub {
    font-family: 'Barlow', sans-serif;
    font-size: 0.85rem;
    font-weight: 300;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #888;
    padding-bottom: 0.3rem;
}

/* ---------- section labels ---------- */
.section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #f5c518;
    margin-bottom: 0.6rem;
}

/* ---------- filter bar ---------- */
.stSelectbox > div, .stTextInput > div > div {
    background: #13131a !important;
    border: 1px solid #2a2a35 !important;
    border-radius: 4px !important;
    color: #e8e6df !important;
}

/* ---------- driver table ---------- */
.driver-table-wrap {
    background: #13131a;
    border: 1px solid #1e1e28;
    border-radius: 6px;
    overflow: hidden;
}
.driver-table-header {
    display: grid;
    grid-template-columns: 2.2rem 1fr 3.5rem 4.5rem 4.5rem 4.5rem 5rem;
    padding: 0.55rem 1rem;
    background: #0d0d14;
    border-bottom: 2px solid #f5c518;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #888;
    gap: 0.5rem;
}
.driver-row {
    display: grid;
    grid-template-columns: 2.2rem 1fr 3.5rem 4.5rem 4.5rem 4.5rem 5rem;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #1a1a22;
    align-items: center;
    cursor: pointer;
    transition: background 0.12s ease;
    gap: 0.5rem;
    font-size: 0.88rem;
}
.driver-row:hover { background: #1c1c27; }
.driver-row.selected {
    background: #1a1a0a;
    border-left: 3px solid #f5c518;
    padding-left: calc(1rem - 3px);
}
.driver-row .pos-badge {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.82rem;
    color: #555;
    text-align: center;
}
.driver-row .d-name {
    font-weight: 500;
    color: #e8e6df;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.driver-row .d-stat {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.9rem;
    color: #aaa;
    text-align: right;
}
.driver-row .d-rating {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    color: #f5c518;
    text-align: right;
}

/* ---------- selected drivers pill row ---------- */
.pill-area-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.5rem;
}
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    min-height: 2.2rem;
    background: #0d0d14;
    border: 1px solid #2a2a35;
    border-radius: 6px;
    padding: 0.5rem 0.6rem;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: #1d1d0d;
    border: 1px solid #f5c518;
    border-radius: 3px;
    padding: 0.18rem 0.5rem;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #f5c518;
    letter-spacing: 0.04em;
}
.pill-empty {
    color: #444;
    font-size: 0.8rem;
    font-style: italic;
    align-self: center;
}

/* ---------- predict button ---------- */
div[data-testid="stButton"] > button {
    background: #f5c518 !important;
    color: #0a0a0f !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.65rem 2.2rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 0 18px rgba(245,197,24,0.25) !important;
}
div[data-testid="stButton"] > button:hover {
    background: #ffd740 !important;
    box-shadow: 0 0 28px rgba(245,197,24,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ---------- results podium ---------- */
.podium-wrap {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-bottom: 2rem;
    padding: 1.5rem 0 0.5rem;
}
.podium-card {
    background: #13131a;
    border: 1px solid #1e1e28;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    min-width: 140px;
    position: relative;
    overflow: hidden;
}
.podium-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.podium-card.p1::before { background: #f5c518; }
.podium-card.p2::before { background: #9e9e9e; }
.podium-card.p3::before { background: #cd7f32; }
.podium-card .podium-pos {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.podium-card.p1 .podium-pos { color: #f5c518; }
.podium-card.p2 .podium-pos { color: #9e9e9e; }
.podium-card.p3 .podium-pos { color: #cd7f32; }
.podium-card .podium-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e8e6df;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.podium-card .podium-score {
    font-size: 0.75rem;
    color: #666;
    margin-top: 0.25rem;
}

/* ---------- full results table ---------- */
.result-row {
    display: grid;
    grid-template-columns: 2.5rem 1fr 5rem 5rem;
    padding: 0.55rem 1rem;
    border-bottom: 1px solid #1a1a22;
    align-items: center;
    font-size: 0.88rem;
    gap: 0.5rem;
}
.result-row.header {
    background: #0d0d14;
    border-bottom: 2px solid #2a2a35;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #555;
}
.result-rank {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 1.1rem;
    color: #f5c518;
    text-align: center;
}
.result-name { font-weight: 500; color: #e8e6df; }
.result-score {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.9rem;
    color: #aaa;
    text-align: right;
}

/* ---------- metric cards ---------- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin-top: 1rem;
}
.metric-card {
    background: #13131a;
    border: 1px solid #1e1e28;
    border-radius: 6px;
    padding: 1rem 1.2rem;
}
.metric-card .m-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.3rem;
}
.metric-card .m-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f5c518;
    line-height: 1;
}
.metric-card .m-sub {
    font-size: 0.72rem;
    color: #555;
    margin-top: 0.2rem;
}

/* ---------- divider ---------- */
.y-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a2a35 30%, #2a2a35 70%, transparent);
    margin: 1.8rem 0;
}

/* ---------- counter badge ---------- */
.counter-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #f5c518;
    color: #0a0a0f;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 0.8rem;
    border-radius: 3px;
    padding: 0.1rem 0.45rem;
    margin-left: 0.4rem;
}
.counter-badge.at-limit { background: #e05c5c; color: #fff; }

/* ---------- hide streamlit chrome ---------- */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_SELECTIONS   = 40
DATA_PATH        = "data/nascar_driver_statistics.csv"
MODEL_PATH       = "models/lgbm_ranker_tuned.pkl"
MODEL_PATH_ORIG  = "models/lgbm_ranker.pkl"
COLUMNS_TO_DROP  = ["id", "Driver", "Points", "Year", "Rank"]

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["Rank"] = df.groupby("Year")["Points"].rank(ascending=True, method="dense")
    df["Rank"] = (df["Rank"] - 1).astype(int)
    df["SeasonKey"] = df["Driver"] + " (" + df["Year"].astype(str) + ")"
    return df

df = load_data()

# ── Session state ──────────────────────────────────────────────────────────────
if "selected_keys" not in st.session_state:
    st.session_state.selected_keys = []
if "results" not in st.session_state:
    st.session_state.results = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nascar-header">
  <div>
    <div class="sub">AI-Powered Season Analysis</div>
    <h1>🏁 NASCAR <span class="accent">Race</span> Predictor</h1>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT — Driver table
# ═══════════════════════════════════════════════════════════════════════════════
with left_col:
    # Filter controls
    fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
    with fcol1:
        search = st.text_input("", placeholder="🔍  Search driver name…", label_visibility="collapsed")
    with fcol2:
        year_opts = ["All Years"] + sorted(df["Year"].unique().tolist(), reverse=True)
        year_filter = st.selectbox("", year_opts, label_visibility="collapsed")
    with fcol3:
        sort_by = st.selectbox("", ["Points ↓", "DriverRating ↓", "Wins ↓", "AvgFinish ↑"], label_visibility="collapsed")

    # Apply filters
    view = df.copy()
    if search:
        view = view[view["Driver"].str.contains(search, case=False, na=False)]
    if year_filter != "All Years":
        view = view[view["Year"] == int(year_filter)]

    sort_map = {
        "Points ↓":       ("Points", False),
        "DriverRating ↓": ("DriverRating", False),
        "Wins ↓":         ("Wins", False),
        "AvgFinish ↑":    ("AvgFinish", True),
    }
    scol, sasc = sort_map[sort_by]
    view = view.sort_values(scol, ascending=sasc).reset_index(drop=True)

    # Count badge
    n_sel = len(st.session_state.selected_keys)
    badge_cls = "counter-badge at-limit" if n_sel >= MAX_SELECTIONS else "counter-badge"
    st.markdown(
        f'<div class="section-label">Driver Roster'
        f'<span class="{badge_cls}">{n_sel} / {MAX_SELECTIONS}</span></div>',
        unsafe_allow_html=True,
    )

    # Table header
    st.markdown("""
    <div class="driver-table-wrap">
      <div class="driver-table-header">
        <span>#</span>
        <span>Driver (Season)</span>
        <span style="text-align:right">Wins</span>
        <span style="text-align:right">Avg Fin</span>
        <span style="text-align:right">Rating</span>
        <span style="text-align:right">Points</span>
        <span style="text-align:right">Year</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Paginate — show 25 rows at a time
    PAGE_SIZE = 25
    total_rows = len(view)
    page_count = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)

    if "table_page" not in st.session_state:
        st.session_state.table_page = 0
    st.session_state.table_page = min(st.session_state.table_page, page_count - 1)

    page_start = st.session_state.table_page * PAGE_SIZE
    page_view  = view.iloc[page_start: page_start + PAGE_SIZE]

    # Render rows as checkboxes inside styled containers
    for i, row in page_view.iterrows():
        key   = row["SeasonKey"]
        is_sel = key in st.session_state.selected_keys
        sel_cls = "selected" if is_sel else ""
        pos_num = page_start + list(page_view.index).index(i) + 1

        col_chk, col_info = st.columns([0.08, 0.92])
        with col_chk:
            checked = st.checkbox("", value=is_sel, key=f"chk_{key}",
                                  label_visibility="collapsed",
                                  disabled=(not is_sel and n_sel >= MAX_SELECTIONS))
        with col_info:
            st.markdown(
                f"""<div class="driver-row {sel_cls}">
                  <span class="pos-badge">{pos_num}</span>
                  <span class="d-name">{row['Driver']} <span style="color:#555;font-size:0.75rem">'{str(row['Year'])[-2:]}</span></span>
                  <span class="d-stat" style="text-align:right">{int(row['Wins'])}</span>
                  <span class="d-stat" style="text-align:right">{row['AvgFinish']:.1f}</span>
                  <span class="d-rating">{row['DriverRating']:.1f}</span>
                  <span class="d-stat" style="text-align:right">{int(row['Points'])}</span>
                  <span class="d-stat" style="text-align:right">{int(row['Year'])}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        # Sync checkbox → session state
        if checked and key not in st.session_state.selected_keys:
            if len(st.session_state.selected_keys) < MAX_SELECTIONS:
                st.session_state.selected_keys.append(key)
                st.rerun()
        elif not checked and key in st.session_state.selected_keys:
            st.session_state.selected_keys.remove(key)
            st.rerun()

    # Pagination controls
    pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
    with pcol1:
        if st.button("← Prev", disabled=st.session_state.table_page == 0):
            st.session_state.table_page -= 1
            st.rerun()
    with pcol2:
        st.markdown(
            f'<p style="text-align:center;color:#555;font-size:0.8rem;padding-top:0.4rem">'
            f'Page {st.session_state.table_page+1} of {page_count} · {total_rows} drivers</p>',
            unsafe_allow_html=True,
        )
    with pcol3:
        if st.button("Next →", disabled=st.session_state.table_page >= page_count - 1):
            st.session_state.table_page += 1
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT — Selected roster + predict + results
# ═══════════════════════════════════════════════════════════════════════════════
with right_col:

    # ── Selected drivers (pills) ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Selected Drivers</div>', unsafe_allow_html=True)

    if st.session_state.selected_keys:
        # Render remove buttons in a compact grid
        cols_per_row = 2
        keys = st.session_state.selected_keys[:]
        for row_start in range(0, len(keys), cols_per_row):
            row_keys = keys[row_start: row_start + cols_per_row]
            pcols = st.columns(cols_per_row)
            for ci, k in enumerate(row_keys):
                with pcols[ci]:
                    label = k if len(k) <= 28 else k[:25] + "…"
                    if st.button(f"✕  {label}", key=f"rm_{k}",
                                 help=f"Remove {k}"):
                        st.session_state.selected_keys.remove(k)
                        st.session_state.results = None
                        st.rerun()

        # Clear all
        if st.button("Clear All", key="clear_all"):
            st.session_state.selected_keys = []
            st.session_state.results = None
            st.rerun()
    else:
        st.markdown(
            '<p style="color:#444;font-size:0.85rem;font-style:italic;padding:0.6rem 0">'
            'No drivers selected yet — click rows in the table.</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="y-divider"></div>', unsafe_allow_html=True)

    # ── Predict button ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)

    predict_disabled = len(st.session_state.selected_keys) < 2
    if predict_disabled:
        st.caption("Select at least 2 drivers to predict rankings.")

    if st.button("⚑  Predict Rankings", disabled=predict_disabled, use_container_width=True):
        # ── Load model ────────────────────────────────────────────────────────
        model = None
        for path in [MODEL_PATH, MODEL_PATH_ORIG]:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    model = pickle.load(f)
                break

        if model is None:
            st.error("No trained model found. Run the training script first.")
        else:
            with st.spinner("Running predictions…"):
                sel_keys = st.session_state.selected_keys
                sel_df   = df[df["SeasonKey"].isin(sel_keys)].copy()

                X = sel_df.drop(columns=COLUMNS_TO_DROP + ["SeasonKey"], errors="ignore")
                sel_df["pred_score"] = model.predict(X)
                sel_df["pred_rank"]  = sel_df["pred_score"].rank(
                    ascending=False, method="first"
                ).astype(int)
                sel_df = sel_df.sort_values("pred_rank")

                # ── Metrics ───────────────────────────────────────────────────
                true_rel = (sel_df["Rank"].max() - sel_df["Rank"]).values
                ndcg_val = ndcg_score([true_rel], [sel_df["pred_score"].values]) \
                           if len(sel_df) >= 2 else np.nan

                # Spearman correlation
                from scipy.stats import spearmanr
                spearman_r, spearman_p = spearmanr(sel_df["Rank"], sel_df["pred_score"])

                # Top-5 accuracy
                actual_top5  = set(sel_df.nsmallest(5, "Rank")["SeasonKey"])
                pred_top5    = set(sel_df.nsmallest(5, "pred_rank")["SeasonKey"])
                top5_acc     = len(actual_top5 & pred_top5) / 5.0

                st.session_state.results = {
                    "df":        sel_df,
                    "ndcg":      ndcg_val,
                    "spearman":  spearman_r,
                    "top5_acc":  top5_acc,
                }

    # ── Results ────────────────────────────────────────────────────────────────
    if st.session_state.results:
        res    = st.session_state.results
        rdf    = res["df"]

        st.markdown('<div class="y-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🏆 Predicted Podium</div>', unsafe_allow_html=True)

        top3 = rdf.head(3)
        pod_classes = ["p1", "p2", "p3"]
        pod_cols = st.columns(3)
        for ci, (_, row) in enumerate(top3.iterrows()):
            with pod_cols[ci]:
                pos_label = ["1ST", "2ND", "3RD"][ci]
                st.markdown(f"""
                <div class="podium-card {pod_classes[ci]}">
                  <div class="podium-pos">{pos_label}</div>
                  <div class="podium-name">{row['Driver']}</div>
                  <div class="podium-score">Score {row['pred_score']:.2f} · {int(row['Year'])}</div>
                </div>""", unsafe_allow_html=True)

        # Full ranking table
        st.markdown('<div class="y-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Full Rankings</div>', unsafe_allow_html=True)

        st.markdown('<div class="driver-table-wrap">', unsafe_allow_html=True)
        st.markdown("""<div class="result-row header">
          <span style="text-align:center">Rank</span>
          <span>Driver (Season)</span>
          <span style="text-align:right">Pred Score</span>
          <span style="text-align:right">Actual Rank</span>
        </div>""", unsafe_allow_html=True)

        for _, row in rdf.iterrows():
            delta = row["pred_rank"] - (row["Rank"] + 1)
            delta_str = (f'<span style="color:#4caf50">▲{abs(int(delta))}</span>' if delta < 0
                         else f'<span style="color:#e05c5c">▼{int(delta)}</span>' if delta > 0
                         else '<span style="color:#888">–</span>')
            st.markdown(f"""<div class="result-row">
              <span class="result-rank">{int(row['pred_rank'])}</span>
              <span class="result-name">{row['Driver']}
                <span style="color:#555;font-size:0.75rem">'{str(int(row['Year']))[-2:]}</span>
                &nbsp;{delta_str}
              </span>
              <span class="result-score">{row['pred_score']:.3f}</span>
              <span class="result-score">{int(row['Rank']) + 1}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Metrics
        st.markdown('<div class="y-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Metrics</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
              <div class="m-label">NDCG Score</div>
              <div class="m-value">{res['ndcg']:.3f}</div>
              <div class="m-sub">Ranking quality</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
              <div class="m-label">Spearman ρ</div>
              <div class="m-value">{res['spearman']:.3f}</div>
              <div class="m-sub">Rank correlation</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
              <div class="m-label">Top-5 Acc</div>
              <div class="m-value">{res['top5_acc']:.0%}</div>
              <div class="m-sub">Correct top 5</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card">
              <div class="m-label">Drivers</div>
              <div class="m-value">{len(rdf)}</div>
              <div class="m-sub">In prediction</div>
            </div>""", unsafe_allow_html=True)