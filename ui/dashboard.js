/* Fraud Risk Intelligence Dashboard -- portfolio presentation layer.
 *
 * Fetches a batch of SYNTHETIC transactions, scored live through the real
 * frozen model artifact, from GET /v1/dashboard/queue -- an additive,
 * local-only endpoint on the existing Stage 14 API (src/api.py). This file
 * renders, filters, sorts, and charts that response; it computes nothing
 * about fraud risk itself. See ui/index.html for the single-transaction
 * demo this dashboard complements (not replaces).
 */
(() => {
  "use strict";

  const API_BASE = "http://127.0.0.1:8000";
  const RISK_BAND_ORDER = ["Low", "Medium", "High", "Critical"];

  const state = {
    raw: null, // last full /v1/dashboard/queue response
    seed: 2024,
    charts: {},
  };

  const el = (id) => document.getElementById(id);

  // Every value interpolated into innerHTML below comes from this page's
  // own /v1/dashboard/queue response, itself built from fixed, closed
  // enumerations server-side (ProductCD/card4/card6/status/risk_band
  // lists) plus numbers it computed -- never free text, and the one real
  // user-text field (the search box) is only ever used for substring
  // filtering, never echoed into HTML. Escaped anyway as defense in depth
  // in case a future change widens any of those fields.
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }

  function fmtMoney(n) {
    return "$" + Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function fmtScore(n) {
    return Number(n).toFixed(4);
  }

  // ---------------------------------------------------------------------
  // Data loading
  // ---------------------------------------------------------------------

  async function fetchJSON(path, params) {
    const url = new URL(API_BASE + path);
    Object.entries(params || {}).forEach(([k, v]) => url.searchParams.set(k, v));
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);
    try {
      const resp = await fetch(url.toString(), { signal: controller.signal });
      clearTimeout(timeoutId);
      return { ok: resp.ok, status: resp.status, body: await resp.json() };
    } catch (err) {
      clearTimeout(timeoutId);
      return { ok: false, status: 0, body: null, networkError: true };
    }
  }

  async function checkHealth() {
    const dot = el("health-dot");
    const text = el("health-text");
    const { ok, body } = await fetchJSON("/health", {});
    if (ok && body && body.artifact_loaded) {
      dot.className = "dot dot-ok";
      text.textContent = "Service healthy";
      el("health-row-status").textContent = "Healthy";
    } else {
      dot.className = "dot dot-bad";
      text.textContent = "Service unavailable";
      el("health-row-status").textContent = "Unavailable";
    }
  }

  async function loadQueue(seed) {
    el("controls-status").textContent = "Scoring batch…";
    const { ok, body, networkError } = await fetchJSON("/v1/dashboard/queue", { n: 48, seed });
    if (!ok || !body || !body.transactions) {
      el("queue-tbody").innerHTML =
        '<tr><td colspan="8" class="table-empty">' +
        (networkError
          ? "Could not reach the API. Start it with <code>.venv/bin/uvicorn src.api:app --host 127.0.0.1 --port 8000</code>, then refresh."
          : "The API could not score a batch right now (model artifact unavailable).") +
        "</td></tr>";
      el("controls-status").textContent = "";
      return;
    }
    state.raw = body;
    state.seed = body.seed;
    populateFilterOptions(body.transactions);
    renderHealthRow(body);
    applyFiltersAndRender();
    el("controls-status").textContent = "Scored " + body.count + " synthetic transactions.";
  }

  function renderHealthRow(body) {
    el("health-row-version").textContent = body.model.artifact_version || "–";
    el("health-row-checksum").textContent = body.model.artifact_sha256
      ? body.model.artifact_sha256.slice(0, 12) + "…"
      : "–";
    el("health-row-contract").textContent = body.model.api_contract_version || "–";
    el("health-row-generated").textContent = new Date(body.generated_at).toLocaleTimeString();
    el("health-row-seed").textContent = body.seed;
  }

  // ---------------------------------------------------------------------
  // Filters
  // ---------------------------------------------------------------------

  function populateFilterOptions(transactions) {
    const productSelect = el("filter-product");
    const cardSelect = el("filter-card-network");
    if (productSelect.dataset.populated) return; // options are stable across refreshes
    const products = [...new Set(transactions.map((t) => t.product_cd))].sort();
    const cards = [...new Set(transactions.map((t) => t.card4))].sort();
    products.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p;
      productSelect.appendChild(opt);
    });
    cards.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      cardSelect.appendChild(opt);
    });
    productSelect.dataset.populated = "1";
  }

  function currentFilters() {
    return {
      product: el("filter-product").value,
      cardNetwork: el("filter-card-network").value,
      band: el("filter-band").value,
      maxAmount: Number(el("filter-amount-max").value),
      search: el("filter-search").value.trim().toLowerCase(),
      sortBy: el("sort-by").value,
    };
  }

  function filteredSorted() {
    if (!state.raw) return [];
    const f = currentFilters();
    let rows = state.raw.transactions.filter((t) => {
      if (f.product && t.product_cd !== f.product) return false;
      if (f.cardNetwork && t.card4 !== f.cardNetwork) return false;
      if (f.band && t.risk_band !== f.band) return false;
      if (t.amount > f.maxAmount) return false;
      if (f.search && !t.transaction_id.toLowerCase().includes(f.search)) return false;
      return true;
    });
    const sorters = {
      risk_score_desc: (a, b) => b.risk_score - a.risk_score,
      risk_score_asc: (a, b) => a.risk_score - b.risk_score,
      amount_desc: (a, b) => b.amount - a.amount,
      amount_asc: (a, b) => a.amount - b.amount,
      id_asc: (a, b) => a.transaction_id.localeCompare(b.transaction_id),
    };
    rows.sort(sorters[f.sortBy] || sorters.risk_score_desc);
    return rows;
  }

  function applyFiltersAndRender() {
    const rows = filteredSorted();
    renderKpis(rows, state.raw);
    renderTable(rows);
    renderCharts(rows, state.raw);
  }

  // ---------------------------------------------------------------------
  // KPIs
  // ---------------------------------------------------------------------

  function renderKpis(filteredRows, raw) {
    const all = raw ? raw.transactions : [];
    const highRisk = all.filter((t) => t.risk_band === "High" || t.risk_band === "Critical").length;
    const reviewQueue = all.filter((t) => t.status !== "Resolved").length;
    const avgScore = all.length ? all.reduce((s, t) => s + t.risk_score, 0) / all.length : 0;
    const maxScore = all.length ? Math.max(...all.map((t) => t.risk_score)) : 0;

    const cards = [
      { label: "Transactions scored", value: all.length, cls: "accent" },
      { label: "High-risk transactions", value: highRisk, cls: "err", sub: highRisk ? Math.round((100 * highRisk) / all.length) + "% of batch" : "" },
      { label: "Review queue size", value: reviewQueue, cls: "warn", sub: "status ≠ Resolved" },
      { label: "Average risk score", value: fmtScore(avgScore), cls: "" },
      { label: "Highest risk score", value: fmtScore(maxScore), cls: "err" },
      { label: "Model artifact version", value: (raw && raw.model.artifact_version) || "–", cls: "" },
      { label: "API / service status", value: raw ? "Online" : "Unknown", cls: raw ? "ok" : "warn" },
    ];

    el("kpi-grid").innerHTML = cards
      .map(
        (c) => `
      <div class="kpi-card">
        <p class="kpi-label">${c.label}</p>
        <p class="kpi-value ${c.cls}">${c.value}</p>
        ${c.sub ? `<p class="kpi-sub">${c.sub}</p>` : ""}
      </div>`
      )
      .join("");
  }

  // ---------------------------------------------------------------------
  // Table
  // ---------------------------------------------------------------------

  function renderTable(rows) {
    const tbody = el("queue-tbody");
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="table-empty">No synthetic transactions match the current filters.</td></tr>';
      return;
    }
    tbody.innerHTML = rows
      .map((t) => {
        const id = escapeHtml(t.transaction_id);
        const band = escapeHtml(t.risk_band);
        return `
      <tr data-id="${id}">
        <td>${id}</td>
        <td>${escapeHtml(t.product_cd)}</td>
        <td>${fmtMoney(t.amount)}</td>
        <td>${fmtScore(t.risk_score)}</td>
        <td><span class="risk-pill ${band}">${band}</span></td>
        <td><span class="status-pill">${escapeHtml(t.status)}</span></td>
        <td>#${t.sequence}</td>
        <td><button type="button" class="ghost inspect-btn" data-id="${id}">Inspect</button></td>
      </tr>`;
      })
      .join("");
    tbody.querySelectorAll("tr[data-id]").forEach((tr) => {
      tr.addEventListener("click", (e) => {
        if (e.target.closest(".inspect-btn")) return; // button handles its own click below
        openDrawer(tr.dataset.id);
      });
    });
    tbody.querySelectorAll(".inspect-btn").forEach((btn) => {
      btn.addEventListener("click", () => openDrawer(btn.dataset.id));
    });
  }

  // ---------------------------------------------------------------------
  // Charts
  // ---------------------------------------------------------------------

  function destroyChart(key) {
    if (state.charts[key]) {
      state.charts[key].destroy();
      delete state.charts[key];
    }
  }

  const CHART_COLORS = {
    Low: "#4caf7d",
    Medium: "#e0a93e",
    High: "#e05a5a",
    Critical: "#ff5c7a",
  };

  function baseOptions(extra) {
    return Object.assign(
      {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#a9adb8" } } },
        scales: {
          x: { ticks: { color: "#a9adb8" }, grid: { color: "#30343d" } },
          y: { ticks: { color: "#a9adb8" }, grid: { color: "#30343d" } },
        },
      },
      extra || {}
    );
  }

  function renderCharts(rows, raw) {
    if (typeof Chart === "undefined") return; // CDN unreachable -- degrade gracefully, table/KPIs still work
    renderDistributionChart(rows);
    renderBandBreakdownChart(rows);
    renderByProductChart(rows);
    renderAmountVsRiskChart(rows);
    renderSequenceChart(rows);
  }

  function renderDistributionChart(rows) {
    if (!rows.length) {
      destroyChart("distribution");
      return;
    }
    const scores = rows.map((t) => t.risk_score);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const span = max - min || 0.0001; // guard against a zero-width range (all identical scores)
    const BIN_COUNT = 10;
    const buckets = new Array(BIN_COUNT).fill(0);
    scores.forEach((s) => {
      const idx = Math.min(BIN_COUNT - 1, Math.floor(((s - min) / span) * BIN_COUNT));
      buckets[idx] += 1;
    });
    el("distribution-note").textContent =
      "Binned across this batch's observed range (" +
      fmtScore(min) +
      "–" +
      fmtScore(max) +
      "), not a fixed 0–1 axis, so the shape stays readable even when scores cluster narrowly.";
    destroyChart("distribution");
    state.charts.distribution = new Chart(el("chart-distribution"), {
      type: "bar",
      data: {
        labels: buckets.map((_, i) => (min + (i / BIN_COUNT) * span).toFixed(3)),
        datasets: [{ label: "Transactions", data: buckets, backgroundColor: "#5aa9ff" }],
      },
      options: baseOptions({ plugins: { legend: { display: false } } }),
    });
  }

  function renderBandBreakdownChart(rows) {
    const counts = RISK_BAND_ORDER.map((band) => rows.filter((t) => t.risk_band === band).length);
    destroyChart("bandBreakdown");
    state.charts.bandBreakdown = new Chart(el("chart-band-breakdown"), {
      type: "doughnut",
      data: {
        labels: RISK_BAND_ORDER,
        datasets: [{ data: counts, backgroundColor: RISK_BAND_ORDER.map((b) => CHART_COLORS[b]) }],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { color: "#a9adb8" } } } },
    });
  }

  function renderByProductChart(rows) {
    const products = [...new Set(rows.map((t) => t.product_cd))].sort();
    const counts = products.map((p) => rows.filter((t) => t.product_cd === p && (t.risk_band === "High" || t.risk_band === "Critical")).length);
    destroyChart("byProduct");
    state.charts.byProduct = new Chart(el("chart-by-product"), {
      type: "bar",
      data: { labels: products, datasets: [{ label: "High + Critical", data: counts, backgroundColor: "#e05a5a" }] },
      options: baseOptions({ plugins: { legend: { display: false } } }),
    });
  }

  function renderAmountVsRiskChart(rows) {
    destroyChart("amountVsRisk");
    state.charts.amountVsRisk = new Chart(el("chart-amount-vs-risk"), {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Transactions",
            data: rows.map((t) => ({ x: t.amount, y: t.risk_score })),
            backgroundColor: rows.map((t) => CHART_COLORS[t.risk_band]),
          },
        ],
      },
      options: baseOptions({
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: "Amount ($)", color: "#a9adb8" }, ticks: { color: "#a9adb8" }, grid: { color: "#30343d" } },
          y: { title: { display: true, text: "Risk score", color: "#a9adb8" }, min: 0, max: 1, ticks: { color: "#a9adb8" }, grid: { color: "#30343d" } },
        },
      }),
    });
  }

  function renderSequenceChart(rows) {
    const bySeq = [...rows].sort((a, b) => a.sequence - b.sequence);
    destroyChart("sequence");
    state.charts.sequence = new Chart(el("chart-sequence"), {
      type: "line",
      data: {
        labels: bySeq.map((t) => "#" + t.sequence),
        datasets: [{ label: "Risk score", data: bySeq.map((t) => t.risk_score), borderColor: "#b183f7", backgroundColor: "rgba(177,131,247,0.15)", tension: 0.25, pointRadius: 2 }],
      },
      options: baseOptions({ plugins: { legend: { display: false } }, scales: { x: { ticks: { color: "#a9adb8", maxTicksLimit: 10 }, grid: { color: "#30343d" } }, y: { min: 0, max: 1, ticks: { color: "#a9adb8" }, grid: { color: "#30343d" } } } }),
    });
  }

  // ---------------------------------------------------------------------
  // Detail drawer
  // ---------------------------------------------------------------------

  function interpretationFor(band, score) {
    const templates = {
      Low: "This transaction's score is low relative to the rest of this batch. It would typically sit at the back of an investigator's queue.",
      Medium: "This transaction's score is moderate relative to the rest of this batch. It may warrant a routine look if reviewer capacity allows.",
      High: "This transaction's score is high relative to the rest of this batch. An investigator would typically prioritize reviewing it sooner.",
      Critical: "This transaction's score is among the highest in this batch. It would typically be at the front of an investigator's review queue.",
    };
    return templates[band] || "";
  }

  function openDrawer(transactionId) {
    if (!state.raw) return;
    const t = state.raw.transactions.find((x) => x.transaction_id === transactionId);
    if (!t) return;

    document.querySelectorAll("#queue-tbody tr").forEach((tr) => tr.classList.toggle("selected", tr.dataset.id === transactionId));

    el("drawer-id").textContent = t.transaction_id;
    el("drawer-score").textContent = fmtScore(t.risk_score);
    const pill = el("drawer-band-pill");
    pill.textContent = t.risk_band;
    pill.className = "risk-pill " + t.risk_band;
    el("drawer-interpretation").textContent = interpretationFor(t.risk_band, t.risk_score);
    el("drawer-product").textContent = t.product_cd;
    el("drawer-card").textContent = (t.card4 || "–") + " / " + (t.card6 || "–");
    el("drawer-amount").textContent = fmtMoney(t.amount);
    el("drawer-status").textContent = t.status;
    el("drawer-sequence").textContent = "#" + t.sequence;

    const features = (state.raw.global_feature_importance || []).slice(0, 8);
    el("drawer-features").innerHTML = features
      .map((f) => `<li><code>${escapeHtml(f.feature)}</code> &mdash; global importance ${f.importance.toFixed(5)}</li>`)
      .join("");

    el("detail-drawer").classList.add("open");
    el("detail-drawer").setAttribute("aria-hidden", "false");
    el("drawer-backdrop").classList.add("open");
  }

  function closeDrawer() {
    el("detail-drawer").classList.remove("open");
    el("detail-drawer").setAttribute("aria-hidden", "true");
    el("drawer-backdrop").classList.remove("open");
    document.querySelectorAll("#queue-tbody tr.selected").forEach((tr) => tr.classList.remove("selected"));
  }

  // ---------------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------------

  function wireControls() {
    ["filter-product", "filter-card-network", "filter-band", "sort-by"].forEach((id) => {
      el(id).addEventListener("change", applyFiltersAndRender);
    });
    el("filter-amount-max").addEventListener("input", () => {
      el("filter-amount-max-value").textContent = fmtMoney(el("filter-amount-max").value).replace(".00", "");
      applyFiltersAndRender();
    });
    el("filter-search").addEventListener("input", applyFiltersAndRender);
    el("reset-filters-btn").addEventListener("click", () => {
      el("filter-product").value = "";
      el("filter-card-network").value = "";
      el("filter-band").value = "";
      el("filter-amount-max").value = 5000;
      el("filter-amount-max-value").textContent = "$5,000";
      el("filter-search").value = "";
      el("sort-by").value = "risk_score_desc";
      applyFiltersAndRender();
    });
    el("refresh-btn").addEventListener("click", () => {
      loadQueue(Math.floor(Math.random() * 1_000_000));
    });
    el("drawer-close").addEventListener("click", closeDrawer);
    el("drawer-backdrop").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeDrawer();
    });
  }

  async function init() {
    wireControls();
    await checkHealth();
    await loadQueue(state.seed);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
