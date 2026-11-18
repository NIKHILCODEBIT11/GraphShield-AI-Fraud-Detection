import time
import yaml
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.models import GATv2FraudDetector

app = FastAPI(
    title="GraphShield Fraud Inference API",
    description="Real-time Graph Attention (GATv2) risk scoring & neighbor attribution service",
    version="1.0.0",
)

with open("configs/config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DATA = torch.load(CFG["paths"]["processed_data"], weights_only=False)
MODEL = GATv2FraudDetector(
    in_channels=DATA.num_features,
    hidden_channels=CFG["model"]["hidden_dim"],
    heads=CFG["model"]["heads"],
    edge_dim=DATA.edge_attr.size(1),
    dropout=CFG["model"]["dropout"],
)
MODEL.load_state_dict(torch.load(CFG["paths"]["best_model"], map_location="cpu", weights_only=True))
MODEL.eval()


class ScoreRequest(BaseModel):
    account_id: int


@app.get("/health")
def health_check():
    return {"status": "healthy", "num_accounts": DATA.num_nodes, "num_edges": DATA.num_edges}


@app.get("/sample-fraud-id")
def get_sample_fraud_id():
    fraud_indices = torch.where((DATA.y == 1) & DATA.test_mask)[0]
    if len(fraud_indices) == 0:
        fraud_indices = torch.where(DATA.y == 1)[0]
    rand_idx = int(torch.randint(0, len(fraud_indices), (1,)).item())
    sample_id = int(fraud_indices[rand_idx].item()) if len(fraud_indices) > 0 else 0
    return {"account_id": sample_id}


@app.post("/score")
def score_account(req: ScoreRequest):
    if req.account_id < 0 or req.account_id >= DATA.num_nodes:
        raise HTTPException(status_code=404, detail="Account ID out of range")

    t0 = time.perf_counter()
    with torch.no_grad():
        logits, (attn_edge_index, attn_weights) = MODEL(
            DATA.x, DATA.edge_index, DATA.edge_attr, return_attention=True
        )
        # Temperature scaling (T=2.0) for calibrated risk probabilities
        prob = float(torch.sigmoid(logits[req.account_id] / 2.0).item())

    dst_mask = attn_edge_index[1] == req.account_id
    neighbor_nodes = attn_edge_index[0][dst_mask]
    neighbor_scores = attn_weights[dst_mask].mean(dim=-1)

    top_neighbors = []
    seen_neighbors = {req.account_id}
    if len(neighbor_scores) > 0:
        sorted_idx = torch.argsort(neighbor_scores, descending=True)
        for idx in sorted_idx.tolist():
            n_id = int(neighbor_nodes[idx].item())
            w = float(neighbor_scores[idx].item())
            if n_id not in seen_neighbors:
                seen_neighbors.add(n_id)
                top_neighbors.append({"neighbor_account_id": n_id, "attention_weight": round(w, 4)})
            if len(top_neighbors) == 4:
                break

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    risk_tier = "HIGH" if prob >= 0.75 else ("MEDIUM" if prob >= 0.40 else "LOW")

    return {
        "account_id": req.account_id,
        "fraud_probability": round(prob, 4),
        "risk_tier": risk_tier,
        "top_influential_neighbors": top_neighbors,
        "inference_latency_ms": latency_ms,
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>GraphShield | GNN Fraud Risk Console</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; background: #0b0f19; color: #e2e8f0; margin: 0; padding: 32px; }
    .container { max-width: 900px; margin: auto; }
    h1 { color: #38bdf8; margin-bottom: 4px; }
    p.sub { color: #94a3b8; margin-top: 0; }
    .controls { display: flex; gap: 12px; margin: 24px 0; }
    input { background: #1e293b; border: 1px solid #334155; color: #fff; padding: 10px 14px; border-radius: 8px; width: 180px; font-size: 15px; }
    button { background: #0284c7; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    button.secondary { background: #334155; }
    button:hover { opacity: 0.9; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 20px; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 13px; }
    .HIGH { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid #ef4444; }
    .MEDIUM { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .LOW { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid #22c55e; }
    svg { width: 100%; height: 260px; background: #0f172a; border-radius: 8px; }
    .metric { font-size: 32px; font-weight: 700; margin: 8px 0; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #1f2937; }
  </style>
</head>
<body>
  <div class="container">
    <h1>GraphShield Risk Console</h1>
    <p class="sub">Real-time GATv2 Fraud Detection & Neighbor Attention Attribution (4,000 Accounts | 54,410 Edges)</p>
    <div class="controls">
      <input type="number" id="accId" placeholder="Account ID (0-3999)" value="0" min="0" max="3999" />
      <button onclick="score()">Analyze Account</button>
      <button class="secondary" onclick="loadFraudSample()">Load Sample Fraud Ring Node</button>
    </div>
    <div class="grid">
      <div class="card">
        <div style="color:#94a3b8;font-size:13px;">FRAUD PROBABILITY</div>
        <div class="metric" id="prob">--</div>
        <span class="badge LOW" id="tier">--</span>
        <div style="margin-top:14px;color:#94a3b8;font-size:13px;">Inference Latency: <strong id="lat" style="color:#e2e8f0;">--</strong></div>
        <table>
          <thead><tr><th>Suspicious Neighbor ID</th><th>GATv2 Attention</th></tr></thead>
          <tbody id="neighborsTable"></tbody>
        </table>
      </div>
      <div class="card">
        <div style="color:#94a3b8;font-size:13px;margin-bottom:8px;">2-HOP EGO ATTENTION GRAPH</div>
        <svg id="egoGraph" viewBox="0 0 400 260"></svg>
      </div>
    </div>
  </div>
  <script>
    async function loadFraudSample() {
      const res = await fetch('/sample-fraud-id');
      const data = await res.json();
      document.getElementById('accId').value = data.account_id;
      score();
    }
    async function score() {
      const id = parseInt(document.getElementById('accId').value);
      const res = await fetch('/score', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({account_id: id})
      });
      const data = await res.json();
      document.getElementById('prob').innerText = (data.fraud_probability * 100).toFixed(2) + '%';
      const tierEl = document.getElementById('tier');
      tierEl.innerText = data.risk_tier + ' RISK';
      tierEl.className = 'badge ' + data.risk_tier;
      document.getElementById('lat').innerText = data.inference_latency_ms + ' ms';

      const tbody = document.getElementById('neighborsTable');
      tbody.innerHTML = data.top_influential_neighbors.map(n =>
        `<tr><td>Account #${n.neighbor_account_id}</td><td>${n.attention_weight}</td></tr>`
      ).join('') || '<tr><td colspan="2">No incoming edges</td></tr>';

      renderGraph(data.account_id, data.risk_tier, data.top_influential_neighbors);
    }
    function renderGraph(centerId, tier, neighbors) {
      const svg = document.getElementById('egoGraph');
      const cx = 200, cy = 130, r = 85;
      const centerColor = tier === 'HIGH' ? '#ef4444' : (tier === 'MEDIUM' ? '#f59e0b' : '#22c55e');
      let html = '';
      neighbors.forEach((n, i) => {
        const angle = (2 * Math.PI * i) / Math.max(neighbors.length, 1);
        const nx = cx + r * Math.cos(angle);
        const ny = cy + r * Math.sin(angle);
        const strokeW = Math.max(1.5, n.attention_weight * 8);
        html += `<line x1="${nx}" y1="${ny}" x2="${cx}" y2="${cy}" stroke="#38bdf8" stroke-width="${strokeW}" stroke-opacity="0.7" />`;
        html += `<circle cx="${nx}" cy="${ny}" r="20" fill="#1e293b" stroke="#38bdf8" stroke-width="2" />`;
        html += `<text x="${nx}" y="${ny+4}" fill="#e2e8f0" font-size="11" text-anchor="middle">#${n.neighbor_account_id}</text>`;
      });
      html += `<circle cx="${cx}" cy="${cy}" r="26" fill="${centerColor}" />`;
      html += `<text x="${cx}" y="${cy+4}" fill="#fff" font-weight="bold" font-size="12" text-anchor="middle">#${centerId}</text>`;
      svg.innerHTML = html;
    }
    score();
  </script>
</body>
</html>"""