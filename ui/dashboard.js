/* Fraud Risk Intelligence Dashboard -- portfolio presentation layer.
 *
 * Fetches a batch of SYNTHETIC transactions, scored live through the real
 * frozen model artifact, from GET /v1/dashboard/queue -- an additive,
 * local-only endpoint on the existing Stage 14 API (src/api.py), or its
 * deliberately bounded public counterpart on src/render_demo.py. This
 * file renders, filters, sorts, and charts that response; it computes
 * nothing about fraud risk itself. See ui/index.html for the
 * single-transaction demo this dashboard complements (not replaces).
 */
(() => {
  "use strict";

  // Local dev serves the UI (python3 -m http.server 8080) and the API
  // (uvicorn, port 8000) from two different origins, so the API base must
  // be hardcoded there. Anywhere else (e.g. a Render deployment serving
  // both the static files and the API from one app) the API is simply
  // "this same origin" -- no build step or server-side text injection
  // needed, unlike ui/index.html's API_BASE, which render_demo.py rewrites
  // at serve time because it is inlined directly in the HTML.
  const IS_LOCAL_DEV_SPLIT_ORIGIN =
    (location.hostname === "127.0.0.1" || location.hostname === "localhost") && location.port === "8080";
  const API_BASE = IS_LOCAL_DEV_SPLIT_ORIGIN ? "http://127.0.0.1:8000" : window.location.origin;

  const RISK_BAND_ORDER = ["Low", "Medium", "High", "Critical"];
  const WORKFLOW_STATUSES = ["New", "In Review", "Escalated", "Resolved"];

  // Fixed demo-batch seeds. These are the SAME values src/render_demo.py
  // hard-codes as its public PUBLIC_DASHBOARD_SEED_ALLOWLIST -- chosen
  // deliberately so this one dropdown behaves identically (not just
  // similarly) whether this page is talking to the local, unrestricted
  // /v1/dashboard/queue or the public, bounded one: on the public
  // service, any seed this page could send is mapped into that same
  // allowlist anyway, so sending these exact values means both backends
  // compute the same underlying batch.
  const DEMO_BATCH_SEEDS = [101, 202, 303, 404, 505];

  const state = {
    raw: null, // last full /v1/dashboard/queue response
    seed: DEMO_BATCH_SEEDS[0],
    sortKey: "risk_score",
    sortDir: "desc",
    selectedId: null, // transaction_id currently shown in the detail drawer, kept in sync across the table AND the Kanban board
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
          ? IS_LOCAL_DEV_SPLIT_ORIGIN
            ? "Could not reach the API. Start it with <code>.venv/bin/uvicorn src.api:app --host 127.0.0.1 --port 8000</code>, then refresh."
            : "Could not reach the scoring service right now. Please try refreshing in a moment."
          : "The API could not score a batch right now (model artifact unavailable).") +
        "</td></tr>";
      el("controls-status").textContent = "";
      return;
    }
    state.raw = body;
    state.seed = body.seed;
    const select = el("batch-select");
    if ([...select.options].some((o) => Number(o.value) === body.seed)) {
      select.value = String(body.seed);
    }
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
    };
  }

  function activeFilterCount(f) {
    let n = 0;
    if (f.product) n += 1;
    if (f.cardNetwork) n += 1;
    if (f.band) n += 1;
    if (f.maxAmount < 5000) n += 1;
    if (f.search) n += 1;
    return n;
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
    const dir = state.sortDir === "asc" ? 1 : -1;
    const key = state.sortKey;
    rows.sort((a, b) => {
      if (key === "transaction_id") return dir * a.transaction_id.localeCompare(b.transaction_id);
      return dir * (a[key] - b[key]);
    });
    return rows;
  }

  function applyFiltersAndRender() {
    const rows = filteredSorted();
    const count = el("filter-count");
    const n = activeFilterCount(currentFilters());
    if (n > 0) {
      count.hidden = false;
      count.textContent = n + " active";
    } else {
      count.hidden = true;
    }
    renderKpis(rows, state.raw);
    renderTable(rows);
    renderCharts(rows, state.raw);
    renderKanban(rows);
  }

  // ---------------------------------------------------------------------
  // KPIs
  // ---------------------------------------------------------------------

  const KPI_ICONS = {
    database:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>',
    alert:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3L2 20h20L12 3z"/><line x1="12" y1="10" x2="12" y2="15"/><circle cx="12" cy="18" r="0.1"/></svg>',
    list:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="20" y2="6"/><line x1="8" y1="12" x2="20" y2="12"/><line x1="8" y1="18" x2="20" y2="18"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/></svg>',
    gauge:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 18a8 8 0 1116 0"/><line x1="12" y1="18" x2="15.5" y2="12.5"/></svg>',
    flame:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3c2 3-2 4-2 7a4 4 0 008 0c0-2-1-3-1-3s1 4-2 4-1-3-3-4c0 0-2 2-2 5a4 4 0 008 0c0-5-4-6-6-9z"/></svg>',
    tag:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 11l8-8h7v7l-8 8a2 2 0 01-3 0l-4-4a2 2 0 010-3z"/><circle cx="15" cy="9" r="1"/></svg>',
    pulse:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12h4l2-7 4 14 2-7h6"/></svg>',
  };

  function renderKpis(filteredRows, raw) {
    const all = raw ? raw.transactions : [];
    // These four describe "what the investigator is currently looking
    // at", so -- like the table, charts, and Kanban board -- they react
    // to the active filters, not the full batch. "Transactions scored"
    // stays batch-wide on purpose: it's a true, filter-independent fact
    // about the real model inference that already happened, not a count
    // of what's currently on screen.
    const scope = filteredRows;
    const isFiltered = scope.length !== all.length;
    const scopeNote = isFiltered ? `of ${scope.length} filtered` : "of full batch";
    const highRisk = scope.filter((t) => t.risk_band === "High" || t.risk_band === "Critical").length;
    const reviewQueue = scope.filter((t) => t.status !== "Resolved").length;
    const avgScore = scope.length ? scope.reduce((s, t) => s + t.risk_score, 0) / scope.length : 0;
    const maxScore = scope.length ? Math.max(...scope.map((t) => t.risk_score)) : 0;

    const cards = [
      {
        label: "Transactions scored",
        value: all.length,
        cls: "accent",
        icon: "database",
        accent: "var(--accent)",
        sub: "real model inferences, this batch",
      },
      {
        label: "High-risk transactions",
        value: highRisk,
        cls: "err",
        icon: "alert",
        accent: "var(--err)",
        sub: scope.length ? Math.round((100 * highRisk) / scope.length) + "% " + scopeNote : scopeNote,
      },
      { label: "Review queue size", value: reviewQueue, cls: "warn", icon: "list", accent: "var(--warn)", sub: "status ≠ Resolved, " + scopeNote },
      { label: "Average risk score", value: fmtScore(avgScore), cls: "", icon: "gauge", accent: "var(--accent-2)", sub: scopeNote },
      { label: "Highest risk score", value: fmtScore(maxScore), cls: "err", icon: "flame", accent: "var(--critical)", sub: scopeNote },
      { label: "Model artifact version", value: (raw && raw.model.artifact_version) || "–", cls: "", icon: "tag", accent: "var(--muted)" },
      { label: "API / service status", value: raw ? "Online" : "Unknown", cls: raw ? "ok" : "warn", icon: "pulse", accent: raw ? "var(--ok)" : "var(--warn)" },
    ];

    el("kpi-grid").innerHTML = cards
      .map(
        (c) => `
      <div class="kpi-card" style="--kpi-accent: ${c.accent};">
        <div class="kpi-card-head">${KPI_ICONS[c.icon]}<p class="kpi-label">${c.label}</p></div>
        <p class="kpi-value ${c.cls}">${c.value}</p>
        ${c.sub ? `<p class="kpi-sub">${c.sub}</p>` : ""}
      </div>`
      )
      .join("");
  }

  // ---------------------------------------------------------------------
  // Table (click-sortable headers)
  // ---------------------------------------------------------------------

  // Applied after every table/Kanban re-render so the open drawer's
  // transaction stays visually highlighted in BOTH views, consistently,
  // even across a filter/sort change that rebuilds their markup from
  // scratch.
  function applySelectionVisuals() {
    document.querySelectorAll("#queue-tbody tr[data-id]").forEach((tr) => {
      tr.classList.toggle("selected", tr.dataset.id === state.selectedId);
    });
    document.querySelectorAll(".kanban-card").forEach((card) => {
      card.classList.toggle("selected", card.dataset.id === state.selectedId);
    });
  }

  function updateSortHeaderIndicators() {
    document.querySelectorAll("#queue-table th.sortable").forEach((th) => {
      if (th.dataset.sortKey === state.sortKey) {
        th.setAttribute("aria-sort", state.sortDir === "asc" ? "ascending" : "descending");
      } else {
        th.setAttribute("aria-sort", "none");
      }
    });
  }

  function renderTable(rows) {
    updateSortHeaderIndicators();
    const tbody = el("queue-tbody");
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="table-empty">No synthetic transactions match the current filters.</td></tr>';
      applySelectionVisuals();
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
    applySelectionVisuals();
  }

  // ---------------------------------------------------------------------
  // Kanban workflow board (read-only grouping of the same filtered rows)
  // ---------------------------------------------------------------------

  function renderKanban(rows) {
    WORKFLOW_STATUSES.forEach((status) => {
      const container = document.querySelector(`[data-cards="${status}"]`);
      const countEl = document.querySelector(`[data-count="${status}"]`);
      const inColumn = rows.filter((t) => t.status === status);
      countEl.textContent = inColumn.length;
      if (!inColumn.length) {
        container.innerHTML = '<p class="kanban-empty">No matching transactions</p>';
        return;
      }
      container.innerHTML = inColumn
        .map(
          (t) => `
        <button type="button" class="kanban-card" data-id="${escapeHtml(t.transaction_id)}">
          <span class="id">${escapeHtml(t.transaction_id)}</span>
          <span class="meta"><span>${fmtMoney(t.amount)}</span><span class="risk-pill ${escapeHtml(t.risk_band)}">${escapeHtml(t.risk_band)}</span></span>
        </button>`
        )
        .join("");
      container.querySelectorAll(".kanban-card").forEach((card) => {
        card.addEventListener("click", () => openDrawer(card.dataset.id));
      });
    });
    applySelectionVisuals();
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

  function toggleBandFilter(band) {
    const sel = el("filter-band");
    sel.value = sel.value === band ? "" : band; // click again to clear
    // Deferred: applyFiltersAndRender() destroys and recreates this chart.
    // Doing that synchronously, inside Chart.js's own click-event dispatch
    // for this same chart, corrupts its internal event-handling state
    // (observed as a real, reproducible "Cannot read properties of
    // undefined (reading 'handleEvent')" exception from chart.umd.min.js).
    // Deferring to the next tick lets Chart.js finish its own event
    // handling first.
    setTimeout(applyFiltersAndRender, 0);
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
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#a9adb8" },
            // A thin slice (e.g. 2 of 48 transactions, ~15 degrees of
            // arc) is a genuinely hard target to click precisely -- this
            // is standard doughnut-chart behavior, not specific to this
            // page (confirmed by testing exact arc-geometry clicks: 3 of
            // 4 bands landed reliably, the smallest did not). The legend
            // swatch is a much larger, reliable target for exactly that
            // case, so it's wired to the SAME filter action instead of
            // Chart.js's default (hiding the slice, which would
            // misrepresent this batch's true composition -- see the
            // "accurately reflect scope" requirement this page follows
            // throughout).
            onClick: (evt, legendItem) => toggleBandFilter(RISK_BAND_ORDER[legendItem.index]),
          },
        },
        onClick: (evt, elements) => {
          if (!elements.length) return;
          toggleBandFilter(RISK_BAND_ORDER[elements[0].index]);
        },
      },
    });
  }

  function renderByProductChart(rows) {
    const products = [...new Set(rows.map((t) => t.product_cd))].sort();
    const counts = products.map((p) => rows.filter((t) => t.product_cd === p && (t.risk_band === "High" || t.risk_band === "Critical")).length);
    destroyChart("byProduct");
    state.charts.byProduct = new Chart(el("chart-by-product"), {
      type: "bar",
      data: { labels: products, datasets: [{ label: "High + Critical", data: counts, backgroundColor: "#e05a5a" }] },
      options: baseOptions({
        plugins: { legend: { display: false } },
        onClick: (evt, elements) => {
          if (!elements.length) return;
          const product = products[elements[0].index];
          const sel = el("filter-product");
          sel.value = sel.value === product ? "" : product;
          setTimeout(applyFiltersAndRender, 0); // see the band-breakdown chart's onClick for why this is deferred
        },
      }),
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

    state.selectedId = transactionId;
    applySelectionVisuals();

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
    state.selectedId = null;
    applySelectionVisuals();
  }

  // ---------------------------------------------------------------------
  // Responsive left navigation
  // ---------------------------------------------------------------------

  function openNav() {
    el("sidenav").classList.add("open");
    el("nav-backdrop").classList.add("open");
    el("nav-toggle").setAttribute("aria-expanded", "true");
  }

  function closeNav() {
    el("sidenav").classList.remove("open");
    el("nav-backdrop").classList.remove("open");
    el("nav-toggle").setAttribute("aria-expanded", "false");
  }

  // ---------------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------------

  function wireControls() {
    ["filter-product", "filter-card-network", "filter-band"].forEach((id) => {
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
      applyFiltersAndRender();
    });

    document.querySelectorAll("#queue-table th.sortable button").forEach((btn) => {
      btn.addEventListener("click", () => {
        const th = btn.closest("th");
        const key = th.dataset.sortKey;
        if (state.sortKey === key) {
          state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
        } else {
          state.sortKey = key;
          state.sortDir = key === "transaction_id" ? "asc" : "desc";
        }
        applyFiltersAndRender();
      });
    });

    el("batch-select").addEventListener("change", (e) => {
      loadQueue(Number(e.target.value));
    });
    el("reload-btn").addEventListener("click", () => {
      loadQueue(state.seed);
    });

    el("drawer-close").addEventListener("click", closeDrawer);
    el("drawer-backdrop").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      closeDrawer();
      closeNav();
    });

    el("nav-toggle").addEventListener("click", openNav);
    el("sidenav-close").addEventListener("click", closeNav);
    el("nav-backdrop").addEventListener("click", closeNav);
    document.querySelectorAll(".sidenav-list a[href^='#']").forEach((link) => {
      link.addEventListener("click", closeNav);
    });
  }

  async function init() {
    wireControls();
    await checkHealth();
    await loadQueue(state.seed);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
