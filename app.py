import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import requests

st.set_page_config(
    page_title="Lead Scoring & Pipeline Manager",
    page_icon="📊",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .priority-high  { background:#EAF3DE; color:#27500A; padding:2px 10px;
                      border-radius:20px; font-size:12px; font-weight:600; }
    .priority-medium{ background:#FpipAEEDA; color:#633806; padding:2px 10px;
                      border-radius:20px; font-size:12px; font-weight:600; }
    .priority-low   { background:#FCEBEB; color:#791F1F; padding:2px 10px;
                      border-radius:20px; font-size:12px; font-weight:600; }
    .trello-card    { background:#F0F4FF; border-left:4px solid #0052CC;
                      border-radius:6px; padding:10px 14px; margin-bottom:8px; }
    .trello-card h4 { margin:0 0 4px 0; font-size:14px; color:#172B4D; }
    .trello-card p  { margin:0; font-size:12px; color:#5E6C84; }
</style>
""", unsafe_allow_html=True)

# ── Trello API helpers ────────────────────────────────────────────────────────
BASE = "https://api.trello.com/1"

def trello_get(path, params):
    r = requests.get(f"{BASE}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def trello_post(path, params):
    r = requests.post(f"{BASE}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def get_boards(key, token):
    return trello_get("/members/me/boards",
                      {"key":key,"token":token,"fields":"id,name"})

def get_lists(board_id, key, token):
    return trello_get(f"/boards/{board_id}/lists",
                      {"key":key,"token":token,"filter":"open","fields":"id,name"})

def get_cards(list_id, key, token):
    return trello_get(f"/lists/{list_id}/cards",
                      {"key":key,"token":token,
                       "fields":"id,name,desc,dateLastActivity,url"})

def create_card(list_id, name, desc, key, token):
    return trello_post("/cards",
                       {"key":key,"token":token,
                        "idList":list_id,"name":name,"desc":desc})

# ── Sample lead data ──────────────────────────────────────────────────────────
SAMPLE_DATA = pd.DataFrame({
    "company":        ["Apex SaaS","FinCore Ltd","MedLogic","RetailHub","NovaTech",
                       "GreenPay","CarePoint","ShopWave","LexPro","BuildCo",
                       "CloudNine","DataPulse","HealthFirst","StyleMart","TradeFin"],
    "industry":       ["SaaS","Finance","Healthcare","Retail","SaaS",
                       "Finance","Healthcare","Retail","Legal","Construction",
                       "SaaS","SaaS","Healthcare","Retail","Finance"],
    "budget_k":       [120,95,80,45,200,60,35,55,75,40,150,90,65,50,110],
    "employees":      [500,200,350,80,1200,150,60,120,180,90,600,300,250,100,400],
    "engagement_score":[88,76,71,54,91,47,32,63,70,41,85,79,58,49,68],
    "deal_stage":     ["Proposal","Demo","Discovery","Contacted","Proposal",
                       "Nurture","Aware","Demo","Discovery","Contacted",
                       "Proposal","Demo","Discovery","Nurture","Proposal"],
    "converted":      [1,1,1,0,1,0,0,1,1,0,1,1,0,0,1],
})

# ── ML model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    le_ind   = LabelEncoder().fit(df["industry"])
    le_stage = LabelEncoder().fit(df["deal_stage"])
    X = pd.DataFrame({
        "budget_k":         df["budget_k"],
        "employees":        df["employees"],
        "engagement_score": df["engagement_score"],
        "industry_enc":     le_ind.transform(df["industry"]),
        "deal_stage_enc":   le_stage.transform(df["deal_stage"]),
    })
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, df["converted"])
    return model, le_ind, le_stage

def score_leads(df, model, le_ind, le_stage):
    df = df.copy()
    df["industry_s"]   = df["industry"].apply(
        lambda x: x if x in le_ind.classes_   else le_ind.classes_[0])
    df["deal_stage_s"] = df["deal_stage"].apply(
        lambda x: x if x in le_stage.classes_ else le_stage.classes_[0])
    X = pd.DataFrame({
        "budget_k":         df["budget_k"],
        "employees":        df["employees"],
        "engagement_score": df["engagement_score"],
        "industry_enc":     le_ind.transform(df["industry_s"]),
        "deal_stage_enc":   le_stage.transform(df["deal_stage_s"]),
    })
    probs = model.predict_proba(X)[:, 1]
    df["conversion_prob"] = (probs * 100).round(1)
    df["priority"] = pd.cut(df["conversion_prob"],
                            bins=[0,40,65,100], labels=["Low","Medium","High"])
    df["strategy"] = df.apply(_strategy, axis=1)
    return df.drop(columns=["industry_s","deal_stage_s"], errors="ignore")

def _strategy(row):
    p, stage = row["conversion_prob"], row.get("deal_stage","")
    if p >= 75: return "Executive outreach + custom proposal"
    if p >= 55: return "Follow-up with ROI calculator" if stage=="Demo" else "Product demo + case study"
    if p >= 35: return "Nurture sequence + trial offer"
    return "Drip email campaign"

# ── Session state init ────────────────────────────────────────────────────────
if "df_scored" not in st.session_state:
    m, li, ls = train_model(SAMPLE_DATA)
    st.session_state.update({"df_scored": score_leads(SAMPLE_DATA,m,li,ls),
                              "model":m, "le_ind":li, "le_stage":ls})
if "trello_connected"    not in st.session_state: st.session_state.trello_connected    = False
if "selected_list_id"    not in st.session_state: st.session_state.selected_list_id    = None
if "selected_list_name"  not in st.session_state: st.session_state.selected_list_name  = "—"
if "chat_history"        not in st.session_state: st.session_state.chat_history        = []

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings & Filters")

    # Trello credentials
    with st.expander("🔑 Trello connection",
                     expanded=not st.session_state.trello_connected):
        st.markdown(
            "1. Get your **API Key** at [trello.com/power-ups/admin](https://trello.com/power-ups/admin)\n"
            "2. Click *'Token'* on that page to generate your **Token**"
        )
        trello_key   = st.text_input("API Key",   type="password")
        trello_token = st.text_input("API Token", type="password")

        if st.button("Connect to Trello"):
            try:
                boards = get_boards(trello_key, trello_token)
                st.session_state.trello_connected = True
                st.session_state.trello_key   = trello_key
                st.session_state.trello_token = trello_token
                st.session_state.boards       = boards
                st.success(f"✅ Connected! {len(boards)} board(s) found.")
            except Exception as e:
                st.error(f"❌ Failed: {e}")

    # Board / list picker
    if st.session_state.trello_connected:
        boards = st.session_state.boards
        chosen_board = st.selectbox("Board", [b["name"] for b in boards])
        board_id = next(b["id"] for b in boards if b["name"]==chosen_board)

        try:
            lists = get_lists(board_id, st.session_state.trello_key,
                              st.session_state.trello_token)
            chosen_list = st.selectbox("List (cards go here)", [l["name"] for l in lists])
            st.session_state.selected_list_id   = next(l["id"]   for l in lists if l["name"]==chosen_list)
            st.session_state.selected_list_name = chosen_list
        except Exception as e:
            st.error(f"Could not fetch lists: {e}")

    st.divider()

    # Lead filters
    uploaded = st.file_uploader("Upload lead CSV", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded)
        required = {"company","industry","budget_k","employees","engagement_score","deal_stage"}
        missing = required - set(raw.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.session_state.df_scored = score_leads(
                raw, st.session_state.model,
                st.session_state.le_ind, st.session_state.le_stage)
            st.success(f"✅ {len(raw)} leads loaded")

    df_all = st.session_state.df_scored
    industries   = st.multiselect("Industry",
                                  sorted(df_all["industry"].unique()),
                                  default=sorted(df_all["industry"].unique()))
    bmin,bmax    = int(df_all["budget_k"].min()), int(df_all["budget_k"].max())
    budget_range = st.slider("Budget ($k)", bmin, bmax, (bmin, bmax))
    prob_min     = st.slider("Min conv. prob (%)", 0, 100, 0, step=5)
    priorities   = st.multiselect("Priority", ["High","Medium","Low"],
                                  default=["High","Medium","Low"])

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_all[
    df_all["industry"].isin(industries) &
    df_all["budget_k"].between(*budget_range) &
    (df_all["conversion_prob"] >= prob_min) &
    df_all["priority"].isin(priorities)
].reset_index(drop=True)

# ── Page header & KPIs ────────────────────────────────────────────────────────
st.title("📊 Lead Scoring & Pipeline Manager")
st.caption(f"Showing **{len(df)}** of **{len(df_all)}** leads")

k1,k2,k3,k4 = st.columns(4)
k1.metric("🔥 High-priority leads", int((df["priority"]=="High").sum()))
k2.metric("🎯 Avg conv. prob",
          f"{df['conversion_prob'].mean():.1f}%" if len(df) else "—")
k3.metric("💰 Total pipeline",   f"${df['budget_k'].sum():,}k")
k4.metric("📦 Avg deal size",
          f"${df['budget_k'].mean():,.0f}k" if len(df) else "—")

st.divider()

# ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([2.2, 1], gap="large")

# ═══════════════ LEFT: Pipeline table ════════════════════════════════════════
with col_left:
    st.subheader("Lead pipeline")

    def color_prob(val):
        if val >= 70: return "color:#3B6D11;font-weight:600"
        if val >= 45: return "color:#BA7517;font-weight:600"
        return "color:#A32D2D;font-weight:600"

    cols    = ["company","industry","budget_k","engagement_score",
               "deal_stage","conversion_prob","priority","strategy"]
    renames = {"company":"Company","industry":"Industry","budget_k":"Budget ($k)",
               "engagement_score":"Engagement","deal_stage":"Stage",
               "conversion_prob":"Conv. Prob (%)","priority":"Priority",
               "strategy":"Strategy"}
    styled = (df[cols].rename(columns=renames)
              .style
              .applymap(color_prob, subset=["Conv. Prob (%)"])
              .format({"Conv. Prob (%)":"{:.1f}","Budget ($k)":"{:,}"}))
    st.dataframe(styled, use_container_width=True, height=300)

    # Push any lead as a Trello card
    st.subheader("📤 Push lead → Trello")
    if not st.session_state.trello_connected:
        st.info("Connect Trello in the sidebar first.")
    elif len(df) == 0:
        st.info("No leads match current filters.")
    else:
        lead_pick = st.selectbox("Choose lead", df["company"], key="lead_pick")
        row = df[df["company"]==lead_pick].iloc[0]
        btn_label = (f"Send to Trello list: "
                     f"**{st.session_state.selected_list_name}**")
        if st.button(btn_label):
            try:
                name = f"[{row['priority']}] {row['company']} — {row['conversion_prob']}% conv."
                desc = (
                    f"Industry: {row['industry']}\n"
                    f"Budget: ${row['budget_k']}k\n"
                    f"Engagement: {row['engagement_score']}\n"
                    f"Stage: {row['deal_stage']}\n"
                    f"Conversion probability: {row['conversion_prob']}%\n"
                    f"Strategy: {row['strategy']}"
                )
                card = create_card(st.session_state.selected_list_id,
                                   name, desc,
                                   st.session_state.trello_key,
                                   st.session_state.trello_token)
                st.success(f"✅ Card created! [Open in Trello ↗]({card['url']})")
                st.session_state.pop("trello_cards_cache", None)   # refresh
            except Exception as e:
                st.error(f"Failed: {e}")

    # Probability chart
    st.subheader("Conversion probability by lead")
    if len(df):
        fig = px.bar(
            df.sort_values("conversion_prob", ascending=False),
            x="company", y="conversion_prob", color="priority",
            color_discrete_map={"High":"#639922","Medium":"#BA7517","Low":"#E24B4A"},
            labels={"company":"","conversion_prob":"Conv. Prob (%)","priority":"Priority"},
            height=240,
        )
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), xaxis_tickangle=-30,
                          legend=dict(orientation="h",yanchor="bottom",y=1.02))
        fig.add_hline(y=65, line_dash="dot", line_color="#185FA5",
                      annotation_text="High threshold")
        fig.add_hline(y=40, line_dash="dot", line_color="#BA7517",
                      annotation_text="Medium threshold")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════ RIGHT: Trello chat panel ════════════════════════════════════
with col_right:
    st.subheader("💬 Trello notes")

    if not st.session_state.trello_connected:
        st.info("🔑 Connect Trello in the sidebar to use this panel.")
    else:
        st.caption(
            f"Messages go to **{st.session_state.selected_list_name}** as Trello cards. "
            "Cards from that list appear below."
        )

        # ── Cards from Trello ─────────────────────────────────────────────────
        c1, c2 = st.columns([3,1])
        c1.markdown("**Cards in Trello**")
        if c2.button("🔄 Refresh"):
            st.session_state.pop("trello_cards_cache", None)

        if "trello_cards_cache" not in st.session_state:
            try:
                cards = get_cards(st.session_state.selected_list_id,
                                  st.session_state.trello_key,
                                  st.session_state.trello_token)
                st.session_state.trello_cards_cache = cards
            except Exception as e:
                st.error(f"Could not load cards: {e}")
                st.session_state.trello_cards_cache = []

        cards = st.session_state.get("trello_cards_cache", [])
        if not cards:
            st.write("No cards yet in this list.")
        else:
            for c in cards[-8:][::-1]:   # newest first, max 8
                preview = c.get("desc","")[:100]
                if len(c.get("desc","")) > 100: preview += "…"
                st.markdown(f"""
<div class="trello-card">
  <h4>🗂 {c['name']}</h4>
  <p>{preview}</p>
  <p><a href="{c.get('url','#')}" target="_blank" style="font-size:11px">Open in Trello ↗</a></p>
</div>""", unsafe_allow_html=True)

        st.divider()

        # ── Chat history (this session) ───────────────────────────────────────
        st.markdown("**This session**")
        chat_box = st.container(height=200)
        with chat_box:
            if not st.session_state.chat_history:
                st.caption("Type a message below — it will appear here and in Trello.")
            for msg in st.session_state.chat_history:
                icon = "🧑" if msg["role"]=="user" else "🟦 Trello"
                st.markdown(f"**{icon}:** {msg['text']}")

        # ── Input form ────────────────────────────────────────────────────────
        with st.form("chat_form", clear_on_submit=True):
            user_msg   = st.text_area("Write a note", height=80,
                                      placeholder="e.g. Follow up with NovaTech by Friday…")
            card_title = st.text_input("Card title (optional)",
                                       placeholder="e.g. NovaTech follow-up task")
            sent = st.form_submit_button("Send to Trello →")

        if sent and user_msg.strip():
            title = card_title.strip() or user_msg.strip()[:60]
            try:
                card = create_card(
                    st.session_state.selected_list_id,
                    title, user_msg.strip(),
                    st.session_state.trello_key,
                    st.session_state.trello_token,
                )
                st.session_state.chat_history.append(
                    {"role":"user",   "text": user_msg.strip()})
                st.session_state.chat_history.append(
                    {"role":"trello", "text": f"Card saved → [{title}]({card['url']})"})
                st.session_state.pop("trello_cards_cache", None)  # auto-refresh
                st.rerun()
            except Exception as e:
                st.error(f"Could not send to Trello: {e}")
