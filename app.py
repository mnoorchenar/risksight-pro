"""
RiskSight Pro — Banking & Insurance Risk Dashboard
Portfolio Demo · Flask · Scikit-learn · Plotly
Deploy on Hugging Face Spaces (Docker) → port 7860
"""

from flask import Flask, render_template_string, jsonify, request
import numpy as np, pandas as pd, json, plotly, warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
app = Flask(__name__)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def dark_layout(fig):
    """Apply consistent dark theme to every Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9", size=11),
        margin=dict(l=8, r=8, t=30, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def rp(title, active, body, scripts=""):
    """Render a full page by injecting body into the shell template."""
    now = datetime.now().strftime("%b %d, %Y  %H:%M")
    html = SHELL.replace("<!-- BODY -->", body).replace("<!-- SCRIPTS -->", scripts)
    return render_template_string(html, title=title, active=active, now=now)

# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA  (realistic enough for a portfolio demo)
# ═══════════════════════════════════════════════════════════════════════════════

N = 1200  # credit records

# ── Credit / Loan ──────────────────────────────────────────────────────────────
cr = pd.DataFrame({
    "age":          np.random.randint(22, 70, N),
    "income":       np.random.lognormal(10.5, .5, N).astype(int),
    "debt_ratio":   np.round(np.random.beta(2, 5, N), 3),
    "credit_score": np.random.randint(300, 850, N),
    "emp_years":    np.random.randint(0, 30, N),
    "loan_amt":     np.random.lognormal(10, .8, N).astype(int),
    "purpose":      np.random.choice(["Mortgage","Auto","Personal","Business"], N,
                                     p=[.35,.25,.25,.15]),
    "region":       np.random.choice(["North","South","East","West"], N),
})
cr["default_prob"] = np.clip(
    .3*(1-(cr.credit_score-300)/550) + .2*cr.debt_ratio +
    .1*(cr.loan_amt/cr.income) + .1*(1-cr.emp_years/30) + np.random.normal(0,.05,N),
    0, 1)
cr["default"] = (cr.default_prob > .3).astype(int)
cr["risk_grade"] = pd.cut(cr.credit_score,[300,580,670,740,800,850],
                           labels=["F","D","C","B","A"])

# ── Fraud / Transactions ───────────────────────────────────────────────────────
NF = 3000
fd = pd.DataFrame({
    "txn_id":    [f"TXN{i:06d}" for i in range(NF)],
    "amount":    np.round(np.random.lognormal(5, 1.5, NF), 2),
    "hour":      np.random.randint(0, 24, NF),
    "merch_risk":np.random.choice(["Low","Medium","High"], NF, p=[.6,.3,.1]),
    "foreign":   np.random.choice([0,1], NF, p=[.85,.15]),
    "velocity":  np.random.randint(1, 20, NF),
    "channel":   np.random.choice(["Online","POS","ATM","Mobile"], NF),
    "date":      pd.date_range("2024-01-01", periods=NF, freq="H"),
})
fd["fraud_prob"] = np.clip(
    .10*(fd.amount>1000).astype(float) + .20*fd.foreign +
    .15*(fd.merch_risk=="High").astype(float) +
    .10*((fd.hour<5)|(fd.hour>22)).astype(float) +
    .05*(fd.velocity>15).astype(float) + np.random.uniform(0,.1,NF), 0, 1)
fd["fraud"] = (fd.fraud_prob > .25).astype(int)

# ── Insurance ──────────────────────────────────────────────────────────────────
NI = 1000
ins = pd.DataFrame({
    "age":         np.random.randint(18, 75, NI),
    "bmi":         np.round(np.random.normal(27, 5, NI), 1),
    "smoker":      np.random.choice([0,1], NI, p=[.75,.25]),
    "region":      np.random.choice(["North","South","East","West"], NI),
    "children":    np.random.randint(0, 5, NI),
    "policy_type": np.random.choice(["Basic","Standard","Premium"], NI, p=[.3,.5,.2]),
    "veh_age":     np.random.randint(0, 20, NI),
})
ins["claim_amt"] = np.round(
    (5000+ins.age*100+ins.bmi*50+ins.smoker*10000+ins.children*500) *
    np.random.lognormal(0,.3,NI), 2)
ins["premium"] = np.round(ins.claim_amt * np.random.uniform(.6,1.4,NI), 2)
ins["loss_ratio"] = np.round(ins.claim_amt/ins.premium, 3)
ins["high_risk"] = ((ins.smoker==1)|(ins.bmi>35)|(ins.age>60)).astype(int)
ins["month"] = np.random.randint(1,13,NI)

# ── Market / Portfolio ─────────────────────────────────────────────────────────
mdt  = pd.date_range(end=datetime.now(), periods=252, freq="B")
mret = np.random.normal(.0003, .012, 252)
mpv  = 10_000_000 * np.cumprod(1+mret)
mkt  = pd.DataFrame({"date":mdt,"ret":mret,"portfolio":mpv})
mkt["drawdown"] = (mkt.portfolio - mkt.portfolio.cummax()) / mkt.portfolio.cummax()

# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN MODELS
# ═══════════════════════════════════════════════════════════════════════════════

# Credit Risk — Random Forest
Xcr = cr[["age","income","debt_ratio","credit_score","emp_years","loan_amt"]]
scr = StandardScaler().fit(Xcr)
mdl_cr = RandomForestClassifier(100, random_state=42).fit(scr.transform(Xcr), cr.default)

# Fraud Detection — Gradient Boosting
Xfd   = pd.concat([fd[["amount","hour","foreign","velocity"]].reset_index(drop=True),
                   pd.get_dummies(fd.merch_risk,prefix="mr").reset_index(drop=True)], axis=1)
FR_COLS = Xfd.columns.tolist()
sfr   = StandardScaler().fit(Xfd)
mdl_fr = GradientBoostingClassifier(100, random_state=42).fit(sfr.transform(Xfd), fd.fraud)

# Underwriting Risk — Logistic Regression
Xins  = pd.concat([ins[["age","bmi","smoker","children","veh_age"]].reset_index(drop=True),
                   pd.get_dummies(ins.region,prefix="r").reset_index(drop=True)], axis=1)
INS_COLS = Xins.columns.tolist()
sins  = StandardScaler().fit(Xins)
mdl_ins = LogisticRegression(random_state=42).fit(sins.transform(Xins), ins.high_risk)

# ═══════════════════════════════════════════════════════════════════════════════
#  SHELL TEMPLATE  (sidebar + topbar, injected with <!-- BODY -->)
# ═══════════════════════════════════════════════════════════════════════════════

SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{{ title }} | RiskSight Pro</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet"/>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    :root{--sw:252px;--bg:#0d1117;--sf:#161b22;--sf2:#21262d;--bd:#30363d;
          --tx:#c9d1d9;--tm:#8b949e;--ac:#00b0ff;--ok:#3fb950;--er:#f85149;--wa:#d29922}
    *{box-sizing:border-box} body{background:var(--bg);color:var(--tx);font-family:'Segoe UI',sans-serif;margin:0}
    /* sidebar */
    .sb{width:var(--sw);height:100vh;position:fixed;left:0;top:0;background:var(--sf);
        border-right:1px solid var(--bd);overflow-y:auto;display:flex;flex-direction:column;z-index:99}
    .brand{padding:16px 18px;border-bottom:1px solid var(--bd)}
    .brand h5{color:var(--ac);margin:0;font-weight:700;font-size:14px;letter-spacing:.5px}
    .brand small{color:var(--tm);font-size:10px}
    .ns{padding:10px 16px 2px;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--tm)}
    .sb .nav-link{color:var(--tm);padding:8px 16px;border-radius:6px;margin:1px 6px;
                  font-size:13px;transition:all .15s;border-left:3px solid transparent;display:flex;align-items:center;gap:9px}
    .sb .nav-link:hover,.sb .nav-link.active{color:#fff;background:rgba(0,176,255,.1);border-left-color:var(--ac)}
    .sb .nav-link i{width:16px;font-size:12px;text-align:center}
    /* main */
    .main{margin-left:var(--sw);min-height:100vh}
    .topbar{background:var(--sf);border-bottom:1px solid var(--bd);padding:13px 22px;
            display:flex;align-items:center;justify-content:space-between}
    .topbar h4{margin:0;font-weight:600;color:#fff;font-size:17px}
    .live{background:rgba(63,185,80,.15);color:var(--ok);border:1px solid var(--ok);
          padding:3px 11px;border-radius:20px;font-size:11px}
    .pb{padding:20px 22px}
    /* kpi card */
    .kpi{background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:16px 18px;transition:transform .2s}
    .kpi:hover{transform:translateY(-2px);border-color:var(--ac)}
    .kpi .lbl{font-size:10px;color:var(--tm);text-transform:uppercase;letter-spacing:1px}
    .kpi .val{font-size:24px;font-weight:700;color:#fff;margin:3px 0}
    .kpi .sub{font-size:11px;color:var(--tm)}
    .kpi .ico{width:40px;height:40px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px}
    /* chart card */
    .cc{background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:18px;margin-bottom:20px}
    .cc h6{color:#fff;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}
    /* table */
    .rt{font-size:12px}
    .rt thead th{background:var(--sf2);color:var(--tm);font-size:10px;text-transform:uppercase;
                 letter-spacing:1px;border:none;padding:9px 11px}
    .rt td{border-color:var(--bd);color:var(--tx);padding:7px 11px;vertical-align:middle}
    .rt tbody tr:hover{background:var(--sf2)}
    /* badges */
    .bl{background:rgba(63,185,80,.15);color:var(--ok);border:1px solid var(--ok);padding:2px 9px;border-radius:20px;font-size:10px}
    .bm{background:rgba(210,153,34,.15);color:var(--wa);border:1px solid var(--wa);padding:2px 9px;border-radius:20px;font-size:10px}
    .bh{background:rgba(248,81,73,.15);color:var(--er);border:1px solid var(--er);padding:2px 9px;border-radius:20px;font-size:10px}
    /* form */
    .fc{background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:22px;margin-bottom:20px}
    .fc h6{color:#fff;font-weight:600;font-size:13px;margin-bottom:16px}
    .form-label{color:var(--tm);font-size:11px;margin-bottom:3px}
    .form-control,.form-select{background:var(--sf2)!important;border:1px solid var(--bd)!important;
                               color:var(--tx)!important;border-radius:7px;font-size:13px}
    .form-control:focus,.form-select:focus{border-color:var(--ac)!important;
                                           box-shadow:0 0 0 3px rgba(0,176,255,.1)!important}
    .form-range::-webkit-slider-thumb{background:var(--ac)}
    .btn-run{background:var(--ac);border:none;border-radius:8px;font-weight:600;color:#000;
             padding:9px 24px;font-size:13px;width:100%}
    .btn-run:hover{background:#33c5ff;color:#000}
    /* result */
    .rb{background:var(--sf2);border-radius:12px;padding:16px;border:1px solid var(--bd)}
    .mbar{height:9px;border-radius:5px;background:var(--bd);overflow:hidden;margin:5px 0}
    .mf{height:100%;border-radius:5px;transition:width .6s}
    /* alert box */
    .alert-dark{background:var(--sf2);border:1px solid var(--bd);color:var(--tx);border-radius:10px}
    ::-webkit-scrollbar{width:5px}
    ::-webkit-scrollbar-thumb{background:var(--bd);border-radius:3px}
  </style>
</head>
<body>
<div class="sb">
  <div class="brand">
    <h5><i class="fas fa-shield-alt me-2"></i>RiskSight Pro</h5>
    <small>Risk Intelligence Platform</small>
  </div>
  <nav class="mt-1 flex-grow-1">
    <div class="ns">Overview</div>
    <a href="/" class="nav-link {{ 'active' if active=='home' }}"><i class="fas fa-th-large"></i>Dashboard</a>
    <div class="ns">Banking Risk</div>
    <a href="/banking/credit-risk"     class="nav-link {{ 'active' if active=='credit'  }}"><i class="fas fa-credit-card"></i>Credit Risk</a>
    <a href="/banking/fraud-detection" class="nav-link {{ 'active' if active=='fraud'   }}"><i class="fas fa-user-secret"></i>Fraud Detection</a>
    <a href="/banking/market-risk"     class="nav-link {{ 'active' if active=='market'  }}"><i class="fas fa-chart-line"></i>Market Risk (VaR)</a>
    <a href="/banking/loan-portfolio"  class="nav-link {{ 'active' if active=='loan'    }}"><i class="fas fa-university"></i>Loan Portfolio</a>
    <div class="ns">Insurance Risk</div>
    <a href="/insurance/claims"        class="nav-link {{ 'active' if active=='claims'  }}"><i class="fas fa-file-medical"></i>Claims Analytics</a>
    <a href="/insurance/underwriting"  class="nav-link {{ 'active' if active=='uw'      }}"><i class="fas fa-clipboard-check"></i>Underwriting Risk</a>
    <a href="/insurance/loss-ratio"    class="nav-link {{ 'active' if active=='loss'    }}"><i class="fas fa-balance-scale"></i>Loss Ratio</a>
  </nav>
  <div class="p-3"><small style="font-size:10px;color:var(--tm)">Portfolio Demo &bull; Synthetic Data<br>RiskSight Pro v1.0 &bull; 2025</small></div>
</div>

<div class="main">
  <div class="topbar">
    <h4>{{ title }}</h4>
    <div class="d-flex align-items-center gap-3">
      <span class="live"><i class="fas fa-circle me-1" style="font-size:8px"></i>Live Demo</span>
      <small style="color:var(--tm);font-size:11px">{{ now }}</small>
    </div>
  </div>
  <div class="pb">
    <!-- BODY -->
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<!-- SCRIPTS -->
</body></html>"""

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE BUILDERS  (small helpers that return chart JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_block(label, value, sub, icon, color):
    return f"""
    <div class="kpi">
      <div class="d-flex justify-content-between align-items-start">
        <div>
          <div class="lbl">{label}</div>
          <div class="val">{value}</div>
          <div class="sub">{sub}</div>
        </div>
        <div class="ico" style="background:rgba({color},.15);color:rgb({color})">{icon}</div>
      </div>
    </div>"""

def plotly_div(div_id, fig_json_str, height=320):
    return f"""
    <div id="{div_id}" style="height:{height}px"></div>
    <script>Plotly.react('{div_id}',{fig_json_str}.data,{fig_json_str}.layout,{{responsive:true,displayModeBar:false}})</script>"""

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTE: HOME DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    total_loans     = len(cr)
    default_rate    = round(cr.default.mean()*100, 1)
    fraud_rate      = round(fd.fraud.mean()*100, 1)
    avg_loss_ratio  = round(ins.loss_ratio.mean(), 3)

    # Chart 1 — default rate by credit grade
    grade_df = cr.groupby("risk_grade", observed=True)["default"].mean().reset_index()
    fig1 = go.Figure(go.Bar(
        x=grade_df.risk_grade.astype(str), y=(grade_df.default*100).round(1),
        marker_color=["#f85149","#d29922","#58a6ff","#3fb950","#00b0ff"],
        text=(grade_df.default*100).round(1).astype(str)+"%", textposition="outside"))
    fig1.update_layout(title="Default Rate by Credit Grade (%)", showlegend=False)

    # Chart 2 — fraud by channel
    ch_df = fd.groupby("channel")["fraud"].mean().reset_index()
    fig2 = px.bar(ch_df, x="channel", y="fraud", color="fraud",
                  color_continuous_scale=["#3fb950","#f85149"], title="Fraud Rate by Channel")

    # Chart 3 — portfolio value
    fig3 = go.Figure(go.Scatter(
        x=mkt.date, y=mkt.portfolio/1e6, fill="tozeroy",
        line=dict(color="#00b0ff",width=2), fillcolor="rgba(0,176,255,.08)"))
    fig3.update_layout(title="Portfolio Value (USD M)", xaxis_title="", yaxis_title="M")

    # Chart 4 — loss ratio by policy type
    lr_df = ins.groupby("policy_type")["loss_ratio"].mean().reset_index()
    fig4 = px.pie(lr_df, names="policy_type", values="loss_ratio",
                  color_discrete_sequence=["#00b0ff","#3fb950","#d29922"],
                  title="Avg Loss Ratio by Policy Type", hole=.45)

    j1,j2,j3,j4 = [dark_layout(f) for f in [fig1,fig2,fig3,fig4]]

    body = f"""
    <!-- KPIs -->
    <div class="row g-3 mb-4">
      <div class="col-md-3">{kpi_block("Total Loan Records", f"{total_loans:,}","Synthetic portfolio","<i class='fas fa-file-invoice-dollar'></i>","0,176,255")}</div>
      <div class="col-md-3">{kpi_block("Avg Default Rate", f"{default_rate}%","Probability of Default (PD)","<i class='fas fa-exclamation-triangle'></i>","248,81,73")}</div>
      <div class="col-md-3">{kpi_block("Fraud Detection Rate", f"{fraud_rate}%","of flagged transactions","<i class='fas fa-user-secret'></i>","210,153,34")}</div>
      <div class="col-md-3">{kpi_block("Avg Loss Ratio", f"{avg_loss_ratio:.2f}","Claims ÷ Premiums","<i class='fas fa-balance-scale'></i>","63,185,80")}</div>
    </div>
    <!-- Charts row 1 -->
    <div class="row g-3 mb-0">
      <div class="col-md-6"><div class="cc"><h6>Credit Risk — PD by Grade</h6>
        <div id="c1" style="height:280px"></div></div></div>
      <div class="col-md-6"><div class="cc"><h6>Fraud Rate by Transaction Channel</h6>
        <div id="c2" style="height:280px"></div></div></div>
    </div>
    <div class="row g-3">
      <div class="col-md-8"><div class="cc"><h6>Portfolio Value (1Y)</h6>
        <div id="c3" style="height:260px"></div></div></div>
      <div class="col-md-4"><div class="cc"><h6>Loss Ratio Mix</h6>
        <div id="c4" style="height:260px"></div></div></div>
    </div>
    <script>
      var fns=[{j1},{j2},{j3},{j4}];
      ["c1","c2","c3","c4"].forEach(function(id,i){{
        Plotly.react(id,fns[i].data,fns[i].layout,{{responsive:true,displayModeBar:false}});
      }});
    </script>"""
    return rp("Risk Dashboard", "home", body)

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTE: CREDIT RISK
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/banking/credit-risk")
def credit_risk():
    # Distribution of credit scores
    fig1 = px.histogram(cr, x="credit_score", nbins=40, color="default",
                        color_discrete_map={0:"#3fb950",1:"#f85149"},
                        barmode="overlay", title="Credit Score Distribution",
                        labels={"default":"Default"})
    # Default prob heatmap by age bucket and debt ratio bucket
    cr["age_grp"]  = pd.cut(cr.age,  [20,30,40,50,60,70], labels=["20s","30s","40s","50s","60s"])
    cr["dr_grp"]   = pd.cut(cr.debt_ratio, [0,.2,.4,.6,.8,1], labels=["0-20%","20-40%","40-60%","60-80%","80-100%"])
    heat = cr.groupby(["age_grp","dr_grp"], observed=True)["default_prob"].mean().unstack()
    fig2 = go.Figure(go.Heatmap(
        z=heat.values, x=heat.columns.astype(str), y=heat.index.astype(str),
        colorscale=[[0,"#3fb950"],[.5,"#d29922"],[1,"#f85149"]], text=heat.values.round(2),
        texttemplate="%{text}", colorbar=dict(title="PD")))
    fig2.update_layout(title="Avg Default Probability — Age vs Debt Ratio")
    # Feature importance
    fi = pd.Series(mdl_cr.feature_importances_,
                   index=["Age","Income","Debt Ratio","Credit Score","Emp Years","Loan Amt"]).sort_values()
    fig3 = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h",
                            marker_color="#00b0ff"))
    fig3.update_layout(title="Feature Importance (Random Forest)")
    j1,j2,j3 = [dark_layout(f) for f in [fig1,fig2,fig3]]

    body = f"""
    <div class="row g-3 mb-3">
      <div class="col-md-3">{kpi_block("Default Rate","{ round(cr.default.mean()*100,1)}%","Probability of Default","<i class='fas fa-times-circle'></i>","248,81,73")}</div>
      <div class="col-md-3">{kpi_block("Avg Credit Score",str(int(cr.credit_score.mean())),"Population average","<i class='fas fa-star'></i>","0,176,255")}</div>
      <div class="col-md-3">{kpi_block("High-Risk Loans",f"{(cr.default_prob>.5).sum():,}","PD > 50%","<i class='fas fa-exclamation-circle'></i>","210,153,34")}</div>
      <div class="col-md-3">{kpi_block("Avg LGD Proxy",f"{ round(cr.debt_ratio.mean()*100,1)}%","Avg debt-to-income ratio","<i class='fas fa-percent'></i>","63,185,80")}</div>
    </div>
    <div class="row g-3">
      <div class="col-lg-5">
        <div class="fc">
          <h6><i class="fas fa-robot me-2" style="color:var(--ac)"></i>Credit Risk Scorer</h6>
          <div class="mb-2">
            <label class="form-label">Age</label>
            <input type="number" class="form-control" id="c_age" value="35" min="18" max="80">
          </div>
          <div class="mb-2">
            <label class="form-label">Annual Income ($)</label>
            <input type="number" class="form-control" id="c_inc" value="60000">
          </div>
          <div class="mb-2">
            <label class="form-label">Debt-to-Income Ratio: <span id="drv">0.30</span></label>
            <input type="range" class="form-range" id="c_dr" min="0" max="1" step=".01" value=".3"
                   oninput="document.getElementById('drv').textContent=parseFloat(this.value).toFixed(2)">
          </div>
          <div class="mb-2">
            <label class="form-label">Credit Score: <span id="csv">680</span></label>
            <input type="range" class="form-range" id="c_cs" min="300" max="850" step="5" value="680"
                   oninput="document.getElementById('csv').textContent=this.value">
          </div>
          <div class="mb-2">
            <label class="form-label">Years Employed</label>
            <input type="number" class="form-control" id="c_ey" value="5" min="0" max="40">
          </div>
          <div class="mb-3">
            <label class="form-label">Loan Amount ($)</label>
            <input type="number" class="form-control" id="c_la" value="25000">
          </div>
          <button class="btn-run" onclick="scoreCr()"><i class="fas fa-brain me-2"></i>Run Credit Assessment</button>
          <div id="cr_res" class="mt-3" style="display:none"></div>
        </div>
      </div>
      <div class="col-lg-7">
        <div class="cc"><h6>Credit Score Distribution</h6><div id="p1" style="height:240px"></div></div>
        <div class="cc"><h6>Feature Importance</h6><div id="p3" style="height:220px"></div></div>
      </div>
    </div>
    <div class="cc"><h6>Default Probability Heatmap — Age Group vs Debt Ratio</h6><div id="p2" style="height:280px"></div></div>
    <script>
      var fns=[{j1},{j2},{j3}];
      ["p1","p2","p3"].forEach(function(id,i){{Plotly.react(id,fns[i].data,fns[i].layout,{{responsive:true,displayModeBar:false}})}});
      function scoreCr(){{
        var payload={{age:+document.getElementById('c_age').value,income:+document.getElementById('c_inc').value,
                      debt_ratio:+document.getElementById('c_dr').value,credit_score:+document.getElementById('c_cs').value,
                      emp_years:+document.getElementById('c_ey').value,loan_amt:+document.getElementById('c_la').value}};
        fetch('/api/credit',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}})
          .then(r=>r.json()).then(d=>{{
            var col=d.pd<.2?'var(--ok)':d.pd<.5?'var(--wa)':'var(--er)';
            var grade=d.pd<.2?'Low Risk':d.pd<.5?'Moderate Risk':'High Risk';
            document.getElementById('cr_res').style.display='block';
            document.getElementById('cr_res').innerHTML=`
              <div class="rb"><div class="d-flex justify-content-between mb-2">
                <span style="color:#fff;font-weight:600">Risk Assessment</span>
                <span style="color:${{col}};font-weight:700">${{grade}}</span></div>
                <div class="lbl mb-1">Probability of Default (PD)</div>
                <div class="mbar"><div class="mf" style="width:${{(d.pd*100).toFixed(0)}}%;background:${{col}}"></div></div>
                <div style="color:${{col}};font-size:22px;font-weight:700">${{(d.pd*100).toFixed(1)}}%</div>
                <hr style="border-color:var(--bd);margin:10px 0">
                <small style="color:var(--tm)">Expected Loss = PD × LGD × EAD</small>
                <div style="color:#fff;font-size:18px;font-weight:600">${{d.expected_loss}}</div>
              </div>`;
          }});
      }}
    </script>"""
    return rp("Credit Risk", "credit", body)

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTE: FRAUD DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/banking/fraud-detection")
def fraud_detection():
    # Fraud by hour
    hr_df = fd.groupby("hour")["fraud"].mean().reset_index()
    fig1 = go.Figure(go.Scatter(x=hr_df.hour, y=hr_df.fraud*100, fill="tozeroy",
                                line=dict(color="#f85149",width=2), fillcolor="rgba(248,81,73,.1)"))
    fig1.update_layout(title="Fraud Rate by Hour of Day (%)")

    # Fraud count by channel
    ch_df = fd.groupby("channel").agg(total=("fraud","count"),fraud=("fraud","sum")).reset_index()
    ch_df["rate"] = ch_df.fraud/ch_df.total
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=ch_df.channel, y=ch_df.total, name="Total", marker_color="#30363d"))
    fig2.add_trace(go.Bar(x=ch_df.channel, y=ch_df.fraud, name="Fraud", marker_color="#f85149"))
    fig2.update_layout(title="Transactions vs Fraud by Channel", barmode="overlay")

    # Fraud amount distribution
    fig3 = px.violin(fd, y="amount", x="fraud", color="fraud",
                     color_discrete_map={0:"#3fb950",1:"#f85149"},
                     title="Transaction Amount Distribution — Legit vs Fraud",
                     labels={"fraud":"Fraud Flag"}, box=True)

    j1,j2,j3 = [dark_layout(f) for f in [fig1,fig2,fig3]]

    # Recent flagged transactions
    recent = fd[fd.fraud==1].sort_values("date",ascending=False).head(12)
    rows = ""
    for _, r in recent.iterrows():
        badge = '<span class="bh">FRAUD</span>' if r.fraud else '<span class="bl">CLEAN</span>'
        mrisk = f'<span class="{"bh" if r.merch_risk=="High" else "bm" if r.merch_risk=="Medium" else "bl"}">{r.merch_risk}</span>'
        rows += f"<tr><td>{r.txn_id}</td><td>${r.amount:,.2f}</td><td>{r.hour:02d}:00</td><td>{mrisk}</td><td>{'Yes' if r.foreign else 'No'}</td><td>{r.channel}</td><td>{badge}</td><td style='color:var(--er)'>{r.fraud_prob:.1%}</td></tr>"

    body = f"""
    <div class="row g-3 mb-3">
      <div class="col-md-3">{kpi_block("Fraud Transactions",str(fd.fraud.sum()),"Detected by model","<i class='fas fa-ban'></i>","248,81,73")}</div>
      <div class="col-md-3">{kpi_block("Fraud Rate",f"{ round(fd.fraud.mean()*100,1)}%","of all transactions","<i class='fas fa-percent'></i>","210,153,34")}</div>
      <div class="col-md-3">{kpi_block("Avg Fraud Amount",f"${fd[fd.fraud==1].amount.mean():,.0f}","vs ${fd[fd.fraud==0].amount.mean():,.0f} clean","<i class='fas fa-dollar-sign'></i>","0,176,255")}</div>
      <div class="col-md-3">{kpi_block("Night Fraud (0-5h)",f"{ round(fd[(fd.hour<5)&(fd.fraud==1)].shape[0]/fd[fd.fraud==1].shape[0]*100,1)}%","of fraud is after hours","<i class='fas fa-moon'></i>","63,185,80")}</div>
    </div>
    <div class="row g-3 mb-2">
      <div class="col-md-5"><div class="cc"><h6>Fraud Rate by Hour</h6><div id="f1" style="height:240px"></div></div></div>
      <div class="col-md-4"><div class="cc"><h6>Channel Breakdown</h6><div id="f2" style="height:240px"></div></div></div>
      <div class="col-md-3"><div class="cc"><h6>Amount Distribution</h6><div id="f3" style="height:240px"></div></div></div>
    </div>
    <div class="cc">
      <h6><i class="fas fa-exclamation-triangle me-2" style="color:var(--er)"></i>Recent Flagged Transactions</h6>
      <div class="table-responsive">
      <table class="table rt">
        <thead><tr><th>TXN ID</th><th>Amount</th><th>Hour</th><th>Merchant Risk</th><th>Foreign</th><th>Channel</th><th>Status</th><th>Fraud Prob</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </div>
    <script>
      var fns=[{j1},{j2},{j3}];
      ["f1","f2","f3"].forEach(function(id,i){{Plotly.react(id,fns[i].data,fns[i].layout,{{responsive:true,displayModeBar:false}})}});
    </script>"""
    return rp("Fraud Detection", "fraud", body)

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTE: MARKET RISK  (VaR / CVaR / Drawdown)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/banking/market-risk")
def market_risk():
    rets = mkt.ret.values
    VaR_95  = -np.percentile(rets, 5)   * 10_000_000
    VaR_99  = -np.percentile(rets, 1)   * 10_000_000
    CVaR_95 = -rets[rets < -VaR_95/10_000_000].mean() * 10_000_000
    vol     = rets.std() * np.sqrt(252)
    sharpe  = (rets.mean()*252) / (rets.std()*np.sqrt(252))
    max_dd  = mkt.drawdown.min()

    # Portfolio time series
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=mkt.date,y=mkt.portfolio/1e6,name="Portfolio",
                              line=dict(color="#00b0ff",width=2),fill="tozeroy",
                              fillcolor="rgba(0,176,255,.06)"))
    fig1.update_layout(title="Portfolio Value (USD M)", yaxis_title="USD M")

    # Return distribution with VaR lines
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=rets*100, nbinsx=60, name="Daily Returns",
                                marker_color="rgba(0,176,255,.6)"))
    fig2.add_vline(x=-VaR_95/10_000_000*100, line_color="#d29922", annotation_text="VaR 95%")
    fig2.add_vline(x=-VaR_99/10_000_000*100, line_color="#f85149", annotation_text="VaR 99%")
    fig2.update_layout(title="Daily Return Distribution (%)")

    # Drawdown
    fig3 = go.Figure(go.Scatter(x=mkt.date, y=mkt.drawdown*100, fill="tozeroy",
                                line=dict(color="#f85149",width=1.5),
                                fillcolor="rgba(248,81,73,.1)"))
    fig3.update_layout(title="Portfolio Drawdown (%)", yaxis_title="%")

    # Rolling VaR (21-day)
    roll_var = pd.Series(rets).rolling(21).apply(lambda x: -np.percentile(x,5)*10_000_000)
    fig4 = go.Figure(go.Scatter(x=mkt.date, y=roll_var/1e3, line=dict(color="#d29922",width=2)))
    fig4.update_layout(title="Rolling 21-day VaR 95% (USD K)")

    j1,j2,j3,j4 = [dark_layout(f) for f in [fig1,fig2,fig3,fig4]]

    body = f"""
    <div class="row g-3 mb-3">
      <div class="col-md-3">{kpi_block("VaR 95% (1-day)",f"${VaR_95:,.0f}","Max daily loss at 95% CI","<i class='fas fa-chart-bar'></i>","210,153,34")}</div>
      <div class="col-md-3">{kpi_block("VaR 99% (1-day)",f"${VaR_99:,.0f}","Max daily loss at 99% CI","<i class='fas fa-exclamation-triangle'></i>","248,81,73")}</div>
      <div class="col-md-3">{kpi_block("CVaR 95%",f"${CVaR_95:,.0f}","Expected Shortfall (ES)","<i class='fas fa-fire'></i>","248,81,73")}</div>
      <div class="col-md-3">{kpi_block("Sharpe Ratio",f"{sharpe:.2f}","Annualized risk-adjusted return","<i class='fas fa-tachometer-alt'></i>","0,176,255")}</div>
    </div>
    <div class="row g-3 mb-2">
      <div class="col-md-8"><div class="cc"><h6>Portfolio Value (1 Year)</h6><div id="m1" style="height:240px"></div></div></div>
      <div class="col-md-4"><div class="cc"><h6>Return Distribution + VaR</h6><div id="m2" style="height:240px"></div></div></div>
    </div>
    <div class="row g-3">
      <div class="col-md-6"><div class="cc"><h6>Drawdown</h6><div id="m3" style="height:230px"></div></div></div>
      <div class="col-md-6"><div class="cc"><h6>Rolling VaR 95%</h6><div id="m4" style="height:230px"></div></div></div>
    </div>
    <div class="alert-dark alert mt-0 mb-0 p-3" style="font-size:12px">
      <i class="fas fa-info-circle me-2" style="color:var(--ac)"></i>
      <b>Basel III Pillar 1:</b> VaR at 99% confidence over 10-day horizon for market risk capital requirement.
      Annualised Volatility: <b>{ round(vol*100,1)}%</b> &bull; Max Drawdown: <b>{ round(max_dd*100,1)}%</b>
    </div>
    <script>
      var fns=[{j1},{j2},{j3},{j4}];
      ["m1","m2","m3","m4"].forEach(function(id,i){{Plotly.react(id,fns[i].data,fns[i].layout,{{responsive:true,displayModeBar:false}})}});
    </script>"""
    return rp("Market Risk — VaR", "market", body)

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTE: LOAN PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/banking/loan-portfolio")
def loan_portfolio():
    # Purpose breakdown
    pur_df = cr.groupby("purpose").agg(count=("loan_amt","count"),total=("loan_amt","sum"),
                                        default_rate=("default","mean")).reset_index()
    fig1 = px.bar(pur_df, x="purpose", y="total", color="default_rate",
                  color_continuous_scale=["#3fb950","#f85149"],
                  title="Loan Volume by Purpose (colored by default rate)",
                  text=pur_df.count.astype(str)+" loans")
    # Risk grade donut
    grade_cnt = cr.risk_grade.value_counts()
    fig2 = go.Figure(go.Pie(labels=grade_cnt.index.astype(str), values=grade_cnt.values,
                            hole=.5, marker_colors=["#f85149","#d29922","#58a6ff","#3fb950","#00b0ff"]))
    fig2.update_layout(title="Portfolio by Credit Grade")
    # Income vs loan scatter
    sample = cr.sample(300)
    fig3 = px.scatter(sample, x="income", y="loan_amt", color="default",
                      color_discrete_map={0:"#3fb950",1:"#f85149"}, opacity=.7,
                      title="Income vs Loan Amount",
                      labels={"default":"Default","income":"Annual Income","loan_amt":"Loan Amount"})
    # Region heatmap
    rg_df = cr.groupby(["region","purpose"])["default"].mean().unstack().fillna(0)
    fig4 = go.Figure(go.Heatmap(z=rg_df.values, x=rg_df.columns, y=rg_df.index,
                                colorscale=[[0,"#3fb950"],[1,"#f85149"]],
                                texttemplate="%{z:.1%}",
                                colorbar=dict(title="Default Rate")))
    fig4.update_layout(title="Default Rate — Region × Loan Purpose")
    j1,j2,j3,j4 = [dark_layout(f) for f in [fig1,fig2,fig3,fig4]]

    body = f"""
    <div class="row g-3 mb-3">
      <div class="col-md-3">{kpi_block("Total Exposure",f"${cr.loan_amt.sum()/1e6:.1f}M","Gross loan book","<i class='fas fa-university'></i>","0,176,255")}</div>
      <div class="col-md-3">{kpi_block("Avg Loan Size",f"${cr.loan_amt.mean():,.0f}","Per borrower","<i class='fas fa-coins'></i>","63,185,80")}</div>
      <div class="col-md-3">{kpi_block("Expected Loss",f"${(cr.default_prob*cr.loan_amt).sum()/1e6:.2f}M","EL = PD × LGD × EAD","<i class='fas fa-times-circle'></i>","248,81,73")}</div>
      <div class="col-md-3">{kpi_block("Concentration Risk",cr.purpose.value_counts().index[0],"Largest loan purpose","<i class='fas fa-layer-group'></i>","210,153,34")}</div>
    </div>
    <div class="row g-3 mb-2">
      <div class="col-md-8"><div class="cc"><h6>Loan Volume by Purpose</h6><div id="l1" style="height:260px"></div></div></div>
      <div class="col-md-4"><div class="cc"><h6>Credit Grade Mix</h6><div id="l2" style="height:260px"></div></div></div>
    </div>
    <div class="row g-3">
      <div class="col-md-6"><div class="cc"><h6>Income vs Loan Amount</h6><div id="l3" style="height:270px"></div></div></div>
      <div class="col-md-6"><div class="cc"><h6>Default Rate Heatmap</h6><div id="l4" style="height:270px"></div></div></div>
    </div>
    <script>
      var fns=[{j1},{j2},{j3},{j4}];
      ["l1","l2","l3","l4"].forEach(function(id,i){{Plotly.react(id,fns[i].data,fns[i].layout,{{responsive:true,displayModeBar:false}})}});
    </script>"""
    return rp("Loan Portfolio", "loan", body)

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTE: CLAIMS ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/insurance/claims")
def claims():
    # Claims by month
    mo_df = ins.groupby("month").agg(count=("claim_amt","count"),total=("claim_amt","sum")).reset_index()
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=mo_df.month, y=mo_df.total/1e3, name="Total Claims ($K)",marker_color="#00b0ff"))
    fig1.add_trace(go.Scatter(x=mo_df.month, y=mo_df.count, name="# Claims",
                              yaxis="y2", line=dict(color="#d29922",width=2), mode="lines+markers"))
    fig1.update_layout(title="Claims Volume by Month",
                       yaxis=dict(title="Total ($K)"),
                       yaxis2=dict(title="# Claims",overlaying="y",side="right",showgrid=False))

    # Claims by policy type
    pt_df = ins.groupby("policy_type")["claim_amt"].describe()[["mean","50%","max"]].reset_index()
    fig2 = go.Figure()
    for col,col_c in [("mean","#00b0ff"),("50%","#3fb950"),("max","#f85149")]:
        fig2.add_trace(go.Bar(x=pt_df.policy_type, y=pt_df[col], name=col.upper(), marker_color=col_c))
    fig2.update_layout(title="Claim Amount Stats by Policy Type", barmode="group")

    # Smoker vs Non-smoker claims
    fig3 = px.box(ins, x="smoker", y="claim_amt", color="smoker",
                  color_discrete_map={0:"#3fb950",1:"#f85149"},
                  title="Claim Amount: Smoker vs Non-Smoker",
                  labels={"smoker":"Smoker (1=Yes)","claim_amt":"Claim ($)"})
    # BMI vs claim
    fig4 = px.scatter(ins.sample(400), x="bmi", y="claim_amt", color="high_risk",
                      color_discrete_map={0:"#3fb950",1:"#f85149"},
                      title="BMI vs Claim Amount (colored by high-risk flag)",
                      trendline="ols", opacity=.65)
    j1,j2,j3,j4 = [dark_layout(f) for f in [fig1,fig2,fig3,fig4]]

    body = f"""
    <div class="row g-3 mb-3">
      <div class="col-md-3">{kpi_block("Total Claims",f"${ins.claim_amt.sum()/1e6:.1f}M","Annual claim exposure","<i class='fas fa-file-medical'></i>","248,81,73")}</div>
      <div class="col-md-3">{kpi_block("Avg Claim",f"${ins.claim_amt.mean():,.0f}","Per policyholder","<i class='fas fa-hand-holding-usd'></i>","0,176,255")}</div>
      <div class="col-md-3">{kpi_block("High-Risk %",f"{ round(ins.high_risk.mean()*100,1)}%","Smokers / BMI>35 / Age>60","<i class='fas fa-heartbeat'></i>","210,153,34")}</div>
      <div class="col-md-3">{kpi_block("Smoker Avg Claim",f"${ins[ins.smoker==1].claim_amt.mean():,.0f}","vs ${ins[ins.smoker==0].claim_amt.mean():,.0f} non-smoker","<i class='fas fa-smoking'></i>","248,81,73")}</div>
    </div>
    <div class="row g-3 mb-2">
      <div class="col-md-8"><div class="cc"><h6>Monthly Claims Volume</h6><div id="cl1" style="height:250px"></div></div></div>
      <div class="col-md-4"><div class="cc"><h6>Claims by Policy Type</h6><div id="cl2" style="height:250px"></div></div></div>
    </div>
    <div class="row g-3">
      <div class="col-md-5"><div class="cc"><h6>Smoker vs Non-Smoker Claims</h6><div id="cl3" style="height:260px"></div></div></div>
      <div class="col-md-7"><div class="cc"><h6>BMI vs Claim Amount</h6><div id="cl4" style="height:260px"></div></div></div>
    </div>
    <script>
      var fns=[{j1},{j2},{j3},{j4}];
      ["cl1","cl2","cl3","cl4"].forEach(function(id,i){{Plotly.react(id,fns[i].data,fns[i].layout,{{responsive:true,displayModeBar:false}})}});
    </script>"""
    return rp("Claims Analytics", "claims", body)

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTE: UNDERWRITING RISK
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/insurance/underwriting")
def underwriting():
    # Feature importances (use coefficients from LogReg)
    feat_names = ["Age","BMI","Smoker","Children","Veh Age","E","N","S","W"][:len(INS_COLS)]
    coef = np.abs(mdl_ins.coef_[0][:len(feat_names)])
    fig1 = go.Figure(go.Bar(x=coef, y=feat_names[:len(coef)], orientation="h",
                            marker_color="#00b0ff"))
    fig1.update_layout(title="Underwriting Risk Factors (|Coefficient|)")

    # Risk score distribution
    probs = mdl_ins.predict_proba(sins.transform(Xins))[:,1]
    fig2 = px.histogram(x=probs, nbins=40, color_discrete_sequence=["#00b0ff"],
                        title="Predicted High-Risk Probability Distribution",
                        labels={"x":"Risk Score"})
    j1,j2 = [dark_layout(f) for f in [fig1,fig2]]

    body = f"""
    <div class="row g-3 mb-3">
      <div class="col-md-3">{kpi_block("High-Risk Policies",str(ins.high_risk.sum()),f"of {NI} total policies","<i class='fas fa-exclamation-circle'></i>","248,81,73")}</div>
      <div class="col-md-3">{kpi_block("Model Accuracy",f"{ round((mdl_ins.predict(sins.transform(Xins))==ins.high_risk.values).mean()*100,1)}%","Logistic Regression","<i class='fas fa-brain'></i>","0,176,255")}</div>
      <div class="col-md-3">{kpi_block("Smoker Risk Premium","+$10K","Additional expected claim","<i class='fas fa-smoking'></i>","210,153,34")}</div>
      <div class="col-md-3">{kpi_block("Obesity (BMI>35)",f"{ round((ins.bmi>35).mean()*100,1)}%","of portfolio","<i class='fas fa-weight'></i>","248,81,73")}</div>
    </div>
    <div class="row g-3">
      <div class="col-lg-4">
        <div class="fc">
          <h6><i class="fas fa-user-shield me-2" style="color:var(--ac)"></i>Underwriting Assessor</h6>
          <div class="mb-2"><label class="form-label">Applicant Age</label>
            <input type="number" class="form-control" id="u_age" value="45"></div>
          <div class="mb-2"><label class="form-label">BMI: <span id="bmiv">27.0</span></label>
            <input type="range" class="form-range" id="u_bmi" min="15" max="50" step=".5" value="27"
                   oninput="document.getElementById('bmiv').textContent=parseFloat(this.value).toFixed(1)"></div>
          <div class="mb-2"><label class="form-label">Smoker</label>
            <select class="form-select" id="u_smk"><option value="0">No</option><option value="1">Yes</option></select></div>
          <div class="mb-2"><label class="form-label">Children</label>
            <input type="number" class="form-control" id="u_ch" value="2" min="0" max="6"></div>
          <div class="mb-2"><label class="form-label">Vehicle Age (years)</label>
            <input type="number" class="form-control" id="u_va" value="5" min="0" max="20"></div>
          <div class="mb-3"><label class="form-label">Region</label>
            <select class="form-select" id="u_reg">
              <option>North</option><option>South</option><option>East</option><option>West</option>
            </select></div>
          <button class="btn-run" onclick="scoreUw()"><i class="fas fa-clipboard-check me-2"></i>Assess Risk</button>
          <div id="uw_res" class="mt-3" style="display:none"></div>
        </div>
      </div>
      <div class="col-lg-8">
        <div class="cc"><h6>Risk Factor Importance</h6><div id="u1" style="height:260px"></div></div>
        <div class="cc"><h6>Risk Score Distribution</h6><div id="u2" style="height:220px"></div></div>
      </div>
    </div>
    <script>
      var fns=[{j1},{j2}];
      ["u1","u2"].forEach(function(id,i){{Plotly.react(id,fns[i].data,fns[i].layout,{{responsive:true,displayModeBar:false}})}});
      function scoreUw(){{
        var payload={{age:+document.getElementById('u_age').value,bmi:+document.getElementById('u_bmi').value,
                      smoker:+document.getElementById('u_smk').value,children:+document.getElementById('u_ch').value,
                      veh_age:+document.getElementById('u_va').value,region:document.getElementById('u_reg').value}};
        fetch('/api/underwriting',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}})
          .then(r=>r.json()).then(d=>{{
            var col=d.risk_score<.3?'var(--ok)':d.risk_score<.6?'var(--wa)':'var(--er)';
            var grade=d.risk_score<.3?'Standard Risk':d.risk_score<.6?'Substandard Risk':'Decline / High Loading';
            document.getElementById('uw_res').style.display='block';
            document.getElementById('uw_res').innerHTML=`
              <div class="rb">
                <div class="d-flex justify-content-between mb-2">
                  <span style="color:#fff;font-weight:600">Underwriting Decision</span>
                  <span style="color:${{col}};font-weight:700">${{grade}}</span></div>
                <div class="lbl mb-1">Risk Score</div>
                <div class="mbar"><div class="mf" style="width:${{(d.risk_score*100).toFixed(0)}}%;background:${{col}}"></div></div>
                <div style="color:${{col}};font-size:22px;font-weight:700">${{(d.risk_score*100).toFixed(1)}}%</div>
                <hr style="border-color:var(--bd);margin:10px 0">
                <div class="row"><div class="col-6"><small class="text-muted">Est. Annual Premium</small>
                  <div style="color:#fff;font-weight:600">${{d.est_premium}}</div></div>
                  <div class="col-6"><small class="text-muted">Risk Loading</small>
                  <div style="color:${{col}};font-weight:600">+${{d.loading}}%</div></div></div>
              </div>`;
          }});
      }}
    </script>"""
    return rp("Underwriting Risk", "uw", body)

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTE: LOSS RATIO
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/insurance/loss-ratio")
def loss_ratio():
    # Loss ratio by region
    rg_df = ins.groupby("region").agg(lr=("loss_ratio","mean"),
                                       claims=("claim_amt","sum"),
                                       premiums=("premium","sum")).reset_index()
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=rg_df.region, y=rg_df.claims/1e3, name="Claims ($K)", marker_color="#f85149"))
    fig1.add_trace(go.Bar(x=rg_df.region, y=rg_df.premiums/1e3, name="Premiums ($K)", marker_color="#3fb950"))
    fig1.update_layout(title="Claims vs Premiums by Region ($K)", barmode="group")

    # Loss ratio by policy type + region heatmap
    lr_heat = ins.groupby(["region","policy_type"])["loss_ratio"].mean().unstack()
    fig2 = go.Figure(go.Heatmap(z=lr_heat.values, x=lr_heat.columns, y=lr_heat.index,
                                colorscale=[[0,"#3fb950"],[.5,"#d29922"],[1,"#f85149"]],
                                texttemplate="%{z:.2f}", zmin=.5, zmax=1.5,
                                colorbar=dict(title="Loss Ratio")))
    fig2.update_layout(title="Loss Ratio Heatmap — Region × Policy Type")

    # Loss ratio distribution
    fig3 = px.histogram(ins, x="loss_ratio", nbins=50, color="policy_type",
                        color_discrete_map={"Basic":"#58a6ff","Standard":"#3fb950","Premium":"#d29922"},
                        title="Loss Ratio Distribution by Policy Type", barmode="overlay", opacity=.7)
    fig3.add_vline(x=1.0, line_color="#f85149", annotation_text="Break-even (LR=1.0)")

    # Combined ratio simulation
    months = list(range(1,13))
    mo_lr = ins.groupby("month")["loss_ratio"].mean()
    expense_ratio = .25
    combined_ratio = mo_lr + expense_ratio
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=months, y=mo_lr.values, name="Loss Ratio",fill="tozeroy",
                              fillcolor="rgba(248,81,73,.1)",line=dict(color="#f85149",width=2)))
    fig4.add_trace(go.Scatter(x=months, y=combined_ratio.values, name="Combined Ratio",
                              line=dict(color="#d29922",width=2,dash="dash")))
    fig4.add_hline(y=1.0, line_color="#3fb950", annotation_text="Profitable threshold")
    fig4.update_layout(title="Loss Ratio vs Combined Ratio by Month")
    j1,j2,j3,j4 = [dark_layout(f) for f in [fig1,fig2,fig3,fig4]]

    body = f"""
    <div class="row g-3 mb-3">
      <div class="col-md-3">{kpi_block("Avg Loss Ratio",f"{ round(ins.loss_ratio.mean(),3)}","<1.0 = profitable","<i class='fas fa-balance-scale'></i>","0,176,255")}</div>
      <div class="col-md-3">{kpi_block("Combined Ratio",f"{ round(ins.loss_ratio.mean()+.25,3)}","LR + Expense Ratio","<i class='fas fa-calculator'></i>","210,153,34")}</div>
      <div class="col-md-3">{kpi_block("Unprofitable Policies",f"{ (ins.loss_ratio>1).sum()}",f"LR>1.0 ({round((ins.loss_ratio>1).mean()*100,1)}% of book)","<i class='fas fa-times'></i>","248,81,73")}</div>
      <div class="col-md-3">{kpi_block("Best Region",rg_df.loc[rg_df.lr.idxmin(),'region'],f"LR = {rg_df.lr.min():.2f}","<i class='fas fa-trophy'></i>","63,185,80")}</div>
    </div>
    <div class="row g-3 mb-2">
      <div class="col-md-7"><div class="cc"><h6>Claims vs Premiums by Region</h6><div id="lr1" style="height:250px"></div></div></div>
      <div class="col-md-5"><div class="cc"><h6>Loss Ratio Distribution</h6><div id="lr3" style="height:250px"></div></div></div>
    </div>
    <div class="row g-3">
      <div class="col-md-6"><div class="cc"><h6>Loss Ratio Heatmap</h6><div id="lr2" style="height:270px"></div></div></div>
      <div class="col-md-6"><div class="cc"><h6>Combined Ratio Trend</h6><div id="lr4" style="height:270px"></div></div></div>
    </div>
    <script>
      var fns=[{j1},{j2},{j3},{j4}];
      ["lr1","lr2","lr3","lr4"].forEach(function(id,i){{Plotly.react(id,fns[i].data,fns[i].layout,{{responsive:true,displayModeBar:false}})}});
    </script>"""
    return rp("Loss Ratio Analysis", "loss", body)

# ═══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS  (called by JS forms)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/credit", methods=["POST"])
def api_credit():
    d   = request.json
    X   = [[d["age"], d["income"], d["debt_ratio"], d["credit_score"], d["emp_years"], d["loan_amt"]]]
    pd_ = float(mdl_cr.predict_proba(scr.transform(X))[0,1])
    lgd = d["debt_ratio"]                   # proxy: debt ratio ≈ LGD
    ead = d["loan_amt"]
    el  = pd_ * lgd * ead
    return jsonify(pd=round(pd_,4), lgd=round(lgd,3), ead=ead,
                   expected_loss=f"${el:,.0f}")

@app.route("/api/fraud", methods=["POST"])
def api_fraud():
    d = request.json
    row = pd.DataFrame([{
        "amount": d["amount"], "hour": d["hour"], "foreign": d["foreign"],
        "velocity": d.get("velocity",1), "mr_High":0,"mr_Low":0,"mr_Medium":0,
    }])
    row[f"mr_{d.get('merch_risk','Low')}"] = 1
    row = row.reindex(columns=FR_COLS, fill_value=0)
    prob = float(mdl_fr.predict_proba(sfr.transform(row))[0,1])
    return jsonify(fraud_prob=round(prob,4), fraud_flag=prob>0.25)

@app.route("/api/underwriting", methods=["POST"])
def api_underwriting():
    d = request.json
    row = pd.DataFrame([{"age":d["age"],"bmi":d["bmi"],"smoker":d["smoker"],
                         "children":d["children"],"veh_age":d["veh_age"],
                         "r_East":0,"r_North":0,"r_South":0,"r_West":0}])
    row[f"r_{d['region']}"] = 1
    row = row.reindex(columns=INS_COLS, fill_value=0)
    prob = float(mdl_ins.predict_proba(sins.transform(row))[0,1])
    base_premium = 5000 + d["age"]*100 + d["bmi"]*50 + d["smoker"]*10000 + d["children"]*500
    loading = round(prob * 80)                     # up to +80% loading
    return jsonify(risk_score=round(prob,4),
                   est_premium=f"${base_premium*(1+loading/100):,.0f}",
                   loading=loading)

# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)