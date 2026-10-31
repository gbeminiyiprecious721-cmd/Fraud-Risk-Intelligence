/* Fraud Risk Intelligence Dashboard -- portfolio presentation layer.
 *
 * Fetches a batch of SYNTHETIC transactions, scored live through the real
 * frozen model artifact, from GET /v1/dashboard/queue -- an additive,
 * local-only endpoint on the existing Stage 14 API (src/api.py), or its
 * deliberately bounded public counterpart on src/render_demo.py. This
 * file renders, filters, sorts, and charts that response; it computes
 * nothing about fraud risk itself. See ui/index.html for the
 * single-transaction demo this dashboard complements (not replaces).
 *
 * Stage 8: this is a single-page app with hash-based client-side routing
 * (#/overview, #/queue, #/analytics, #/workflow, #/insights,
 * #/responsible-ai) across six views, all sharing ONE `state` object as
 * the single source of truth (current demo batch, selected transaction,
 * filters). Hash routing was chosen deliberately over path-based routing
 * with a server fallback: it behaves identically in local dev (plain
 * `python3 -m http.server`, no server changes possible) and on Render,
 * and a hard refresh or direct link always works with zero server code,
 * where path-based routing would need a new FastAPI catch-all that local
 * dev's static server can't replicate.
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
  const VIEWS = ["overview", "queue", "analytics", "workflow", "insights", "responsible-ai", "capacity", "benchmark", "monitoring"];
  const DEFAULT_VIEW = "overview";

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
    selectedId: null, // transaction_id currently shown in the detail drawer -- a SINGLE source of truth shared by every table/Kanban/preview across every view, deliberately never cleared on navigation
    currentView: DEFAULT_VIEW,
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
  // Router (hash-based: works identically in local dev and on Render,
  // and for a hard refresh or a bookmarked/shared direct link, with zero
  // server-side routes beyond serving this one document)
  // ---------------------------------------------------------------------

  function parseViewFromHash() {
    const raw = location.hash.replace(/^#\/?/, "");
    return VIEWS.includes(raw) ? raw : DEFAULT_VIEW;
  }

  function renderActiveView() {
    const view = parseViewFromHash();
    state.currentView = view;
    const historical = ["capacity", "benchmark", "monitoring"].includes(view);
    document.body.classList.toggle("capacity-active", historical);
    document.body.classList.toggle("benchmark-active", view === "benchmark");
    document.body.classList.toggle("case-workspace-active", ["queue","workflow"].includes(view));
    const populationBadge = document.querySelector(".topbar .badge-synthetic");
    populationBadge.textContent = historical ? "Historical validation / portfolio" : "Synthetic / portfolio demonstration";
    populationBadge.title = historical
      ? "Historical validation aggregates, separate from the synthetic demo batch; monitoring uses no outcomes."
      : "Every transaction shown here is fabricated for this demo, never a real or training record.";
    document.querySelectorAll(".view").forEach((v) => {
      v.classList.toggle("active", v.dataset.view === view);
    });
    document.querySelectorAll(".sidenav-list a[data-view]").forEach((a) => {
      const isActive = a.dataset.view === view;
      a.classList.toggle("active", isActive);
      if (isActive) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    closeNav();
    const container = el("view-container");
    if (container) container.scrollTop = 0;
    // Chart.js sizes a canvas against its container at creation time; a
    // canvas inside a `display:none` view at that moment can end up
    // wrongly sized. Re-rendering here, after the view's `display` has
    // just been switched to visible, guarantees every chart is (re)built
    // while its canvas is actually visible and correctly sized -- simpler
    // and more robust than trying to special-case "only render the active
    // view's charts" bookkeeping.
    if (state.raw) applyFiltersAndRender();
    const heading = document.getElementById("view-heading-" + view);
    if (heading) heading.focus();
    if (view === "capacity" && window.CapacitySimulator) window.CapacitySimulator.show();
    if (view === "benchmark" && window.ModelBenchmark) window.ModelBenchmark.show();
    if (view === "monitoring" && window.ModelMonitoring) window.ModelMonitoring.show();
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
      const message = networkError
        ? IS_LOCAL_DEV_SPLIT_ORIGIN
          ? "Could not reach the API. Start it with <code>.venv/bin/uvicorn src.api:app --host 127.0.0.1 --port 8000</code>, then refresh."
          : "Could not reach the scoring service right now. Please try refreshing in a moment."
        : "The API could not score a batch right now (model artifact unavailable).";
      el("queue-tbody").innerHTML = `<tr><td colspan="8" class="table-empty">${message}</td></tr>`;
      if (el("preview-tbody")) el("preview-tbody").innerHTML = `<tr><td colspan="6" class="table-empty">${message}</td></tr>`;
      el("controls-status").textContent = "";
      return;
    }
    closeDrawer();
    state.raw = body;
    window.CaseUI.setBatch(body, applyFiltersAndRender);
    state.seed = body.seed;
    const select = el("batch-select");
    if ([...select.options].some((o) => Number(o.value) === body.seed)) {
      select.value = String(body.seed);
    }
    populateFilterOptions(body.transactions);
    renderHealthRow(body);
    applyFiltersAndRender(); // also sets #controls-status / #global-scope-status via updateScopeStatus()
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
  // Filters (global: one set of filter state affects every view, per the
  // explicit requirement that KPI/chart/queue counts never diverge --
  // Overview's preview, Analytics' charts, and the Workflow board all
  // read from the SAME filteredSorted() result as the Queue table.)
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
    for (const id of ["case-status-filter","case-analyst-filter","case-priority-filter"]) if (el(id).value) n += 1;
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
      if (f.search && !t.transaction_id.toLowerCase().includes(f.search) && !t.transaction_id.replace("TXN-","CASE-").toLowerCase().includes(f.search)) return false;
      if (!window.CaseUI.filter(t)) return false;
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
    updateScopeStatus(rows, n);
    renderKpis(rows, state.raw);
    renderTable(rows);
    renderPreviewTable(rows);
    renderCharts(rows, state.raw);
    renderKanban(rows);
    window.CaseUI.workload(rows);
    renderInsightsFeatureImportance();
  }

  // Stage 15: a single, always-visible (every view, not just Queue)
  // statement of how much of the batch is currently on screen, plus
  // making the Reset button visually come alive only once there's
  // something for it to actually do.
  function updateScopeStatus(rows, activeCount) {
    const total = state.raw ? state.raw.transactions.length : 0;
    const scopeEl = el("global-scope-status");
    const resetBtn = el("reset-filters-btn");
    if (scopeEl) {
      if (activeCount > 0) {
        scopeEl.textContent = `Showing ${rows.length} of ${total} transactions · ${activeCount} filter${activeCount === 1 ? "" : "s"} active`;
        scopeEl.classList.add("filtered");
      } else {
        scopeEl.textContent = total ? `Showing all ${total} transactions` : "";
        scopeEl.classList.remove("filtered");
      }
    }
    if (resetBtn) resetBtn.classList.toggle("has-active-filters", activeCount > 0);
    const statusEl = el("controls-status");
    if (statusEl && state.raw) {
      statusEl.textContent =
        activeCount > 0
          ? `Showing ${rows.length} of ${total} synthetic transactions (${activeCount} filter${activeCount === 1 ? "" : "s"} active).`
          : `Scored ${total} synthetic transactions.`;
    }
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
  // Table (click-sortable headers) + Overview preview table
  // ---------------------------------------------------------------------

  // Applied after every table/preview/Kanban re-render so the open
  // drawer's transaction stays visually highlighted everywhere it
  // appears, consistently, even across a filter/sort/view change that
  // rebuilds their markup from scratch.
  function applySelectionVisuals() {
    document.querySelectorAll("#queue-tbody tr[data-id]").forEach((tr) => {
      tr.classList.toggle("selected", tr.dataset.id === state.selectedId);
    });
    document.querySelectorAll("#preview-tbody tr[data-id]").forEach((tr) => {
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

  function queueRowHtml(t, columns) {
    const id = escapeHtml(t.transaction_id);
    const band = escapeHtml(t.risk_band);
    const caseStatus = window.CaseUI.get(t.transaction_id)?.status || "Not opened";
    const cells = {
      id: `<td>${id}</td>`,
      product: `<td>${escapeHtml(t.product_cd)}</td>`,
      amount: `<td>${fmtMoney(t.amount)}</td>`,
      score: `<td>${fmtScore(t.risk_score)}</td>`,
      band: `<td><span class="risk-pill ${band}">${band}</span></td>`,
      status: `<td><span class="status-pill" data-status="${escapeHtml(caseStatus)}" title="${caseStatus === "Not opened" ? "Seeded display label only; no investigation case or analyst decision yet." : "Actual investigation case status, saved in this browser."}">${escapeHtml(window.CaseUI.label(t))}</span></td>`,
      sequence: `<td>#${t.sequence}</td>`,
      inspect: `<td><button type="button" class="ghost inspect-btn" data-id="${id}">Inspect</button></td>`,
    };
    return `<tr data-id="${id}">${columns.map((c) => cells[c]).join("")}</tr>`;
  }

  function renderTable(rows) {
    updateSortHeaderIndicators();
    const tbody = el("queue-tbody");
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="table-empty">${escapeHtml(window.CaseUI.emptyMessage())}</td></tr>`;
      applySelectionVisuals();
      return;
    }
    tbody.innerHTML = rows
      .map((t) => queueRowHtml(t, ["id", "product", "amount", "score", "band", "status", "sequence", "inspect"]))
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

  function renderPreviewTable(rows) {
    const tbody = el("preview-tbody");
    if (!tbody) return;
    const PREVIEW_LIMIT = 8;
    const preview = rows.slice(0, PREVIEW_LIMIT);
    if (!preview.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="table-empty">${escapeHtml(window.CaseUI.emptyMessage())}</td></tr>`;
      applySelectionVisuals();
      return;
    }
    tbody.innerHTML = preview.map((t) => queueRowHtml(t, ["id", "product", "amount", "score", "band", "status"])).join("");
    tbody.querySelectorAll("tr[data-id]").forEach((tr) => {
      tr.addEventListener("click", () => openDrawer(tr.dataset.id));
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
      if (!container || !countEl) return;
      const inColumn = rows.filter((t) => window.CaseUI.status(t) === status);
      countEl.textContent = inColumn.length;
      if (!inColumn.length) {
        container.innerHTML = `<p class="kanban-empty">${rows.length ? 'No matching transactions in this lane.' : escapeHtml(window.CaseUI.emptyMessage())}</p>`;
        return;
      }
      container.innerHTML = inColumn
        .map(
          (t) => `
        <button type="button" class="kanban-card" data-id="${escapeHtml(t.transaction_id)}">
          <span class="id">${escapeHtml(t.transaction_id)}</span><span class="case-card-meta">${escapeHtml(window.CaseUI.label(t))}</span>
          <span class="meta"><span>${fmtMoney(t.amount)}</span><span class="risk-pill ${escapeHtml(t.risk_band)}">${escapeHtml(t.risk_band)}</span></span>
        </button>`
        )
        .join("");
      container.querySelectorAll(".kanban-card").forEach((card) => {
        card.addEventListener("click", () => openDrawer(card.dataset.id));
      });
    });
    applySelectionVisuals();
    updateKanbanScrollAffordance();
  }

  // Stage 10: below 901px the 4-column board scrolls horizontally and only
  // ~1 column fits in view. This gives a real, scroll-position-aware cue
  // (edge fades + dots) instead of leaving users to discover it by luck.
  // Re-run on every re-render (above) and on scroll/resize (wired below) --
  // cheap DOM reads, no measurable cost at this scale (4 columns).
  function updateKanbanScrollAffordance() {
    const board = el("kanban-board");
    const wrap = el("kanban-scroll-wrap");
    const dots = document.querySelectorAll("#kanban-scroll-dots .dot");
    if (!board || !wrap) return;
    const maxScroll = board.scrollWidth - board.clientWidth;
    const atStart = board.scrollLeft <= 2;
    const atEnd = maxScroll <= 2 || board.scrollLeft >= maxScroll - 2;
    wrap.dataset.atStart = String(atStart);
    wrap.dataset.atEnd = String(atEnd);
    if (!dots.length) return;
    const columns = board.querySelectorAll(".kanban-column");
    if (!columns.length) return;
    const viewportCenter = board.scrollLeft + board.clientWidth / 2;
    let activeIndex = 0;
    let bestDistance = Infinity;
    columns.forEach((col, i) => {
      const colCenter = col.offsetLeft + col.offsetWidth / 2;
      const distance = Math.abs(colCenter - viewportCenter);
      if (distance < bestDistance) {
        bestDistance = distance;
        activeIndex = i;
      }
    });
    dots.forEach((dot, i) => dot.classList.toggle("active", i === activeIndex));
  }

  // ---------------------------------------------------------------------
  // Model Insights: global feature importance (Stage 11, model-level)
  // ---------------------------------------------------------------------

  function renderInsightsFeatureImportance() {
    const container = el("insights-feature-importance");
    if (!container || !state.raw) return;
    const features = (state.raw.global_feature_importance || []).slice(0, 15);
    // A small horizontal bar per row, scaled against the top (already
    // highest-ranked) entry -- the same importance values already shown
    // as text, visualized, not a new or re-derived data source.
    const maxImportance = features.length ? features[0].importance : 1;
    container.innerHTML = features
      .map((f, i) => {
        const widthPct = maxImportance > 0 ? Math.max(4, Math.round((f.importance / maxImportance) * 100)) : 0;
        return `
      <li>
        <div class="feature-row">
          <span><span class="feature-rank">${i + 1}.</span> <code>${escapeHtml(f.feature)}</code></span>
          <span class="feature-value">${f.importance.toFixed(5)}</span>
        </div>
        <div class="feature-bar-track"><div class="feature-bar-fill" style="width: ${widthPct}%"></div></div>
      </li>`;
      })
      .join("");
  }

  // ---------------------------------------------------------------------
  // Charts (full versions on Analytics; compact non-interactive previews
  // on Overview -- both driven by the same parameterized functions so
  // there is exactly one place that computes each chart's data)
  // ---------------------------------------------------------------------

  function destroyChart(key) {
    if (state.charts[key]) {
      state.charts[key].destroy();
      delete state.charts[key];
    }
  }

  // Stage 15: a chart that previously just vanished (destroyed, blank
  // canvas) or rendered with all-zero data (a visually empty donut/bars)
  // when the current filters match nothing now shows an explicit message
  // instead -- consistent with the table/Kanban board's existing empty
  // states. `canvasId` + "-empty" is this file's own naming convention
  // for each chart's paired empty-state element (see dashboard.html).
  function toggleChartEmptyState(canvasId, isEmpty) {
    const empty = el(canvasId + "-empty");
    const canvas = el(canvasId);
    if (empty) empty.hidden = !isEmpty;
    if (canvas) canvas.style.visibility = isEmpty ? "hidden" : "visible";
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
    renderDistributionChart(rows, { canvasId: "chart-distribution", chartKey: "distribution", noteId: "distribution-note" });
    renderBandBreakdownChart(rows, { canvasId: "chart-band-breakdown", chartKey: "bandBreakdown", interactive: true });
    renderByProductChart(rows);
    renderAmountVsRiskChart(rows);
    renderSequenceChart(rows);
    // Overview's compact previews: same data, smaller/non-interactive
    // canvases. Deliberately read-only (no legend, no click handlers) --
    // the Analytics view is the one dedicated, clickable home for these,
    // and a silently-interactive mini chart tucked into a dense overview
    // would be a hidden, undiscoverable control.
    renderDistributionChart(rows, { canvasId: "chart-distribution-mini", chartKey: "distributionMini" });
    renderBandBreakdownChart(rows, { canvasId: "chart-band-breakdown-mini", chartKey: "bandBreakdownMini", interactive: false });
  }

  function renderDistributionChart(rows, opts) {
    opts = opts || {};
    const canvas = el(opts.canvasId || "chart-distribution");
    const chartKey = opts.chartKey || "distribution";
    if (!canvas) return;
    toggleChartEmptyState(opts.canvasId || "chart-distribution", !rows.length);
    if (!rows.length) {
      destroyChart(chartKey);
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
    if (opts.noteId) {
      const noteEl = el(opts.noteId);
      if (noteEl) {
        noteEl.textContent =
          "Binned across this batch's observed range (" +
          fmtScore(min) +
          "–" +
          fmtScore(max) +
          "), not a fixed 0–1 axis, so the shape stays readable even when scores cluster narrowly.";
      }
    }
    destroyChart(chartKey);
    state.charts[chartKey] = new Chart(canvas, {
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

  function renderBandBreakdownChart(rows, opts) {
    opts = opts || {};
    const canvas = el(opts.canvasId || "chart-band-breakdown");
    const chartKey = opts.chartKey || "bandBreakdown";
    const interactive = opts.interactive !== false;
    const counts = RISK_BAND_ORDER.map((band) => rows.filter((t) => t.risk_band === band).length);
    destroyChart(chartKey);
    if (!canvas) return;
    toggleChartEmptyState(opts.canvasId || "chart-band-breakdown", !rows.length);
    if (!rows.length) return;
    const legendConfig = interactive
      ? {
          position: "bottom",
          labels: { color: "#a9adb8" },
          // A thin slice (e.g. 2 of 48 transactions, ~15 degrees of arc)
          // is a genuinely hard target to click precisely -- standard
          // doughnut-chart behavior, not specific to this page (confirmed
          // by testing exact arc-geometry clicks: 3 of 4 bands landed
          // reliably, the smallest did not). The legend swatch is a much
          // larger, reliable target for exactly that case, so it's wired
          // to the SAME filter action instead of Chart.js's default
          // (hiding the slice, which would misrepresent this batch's true
          // composition).
          onClick: (evt, legendItem) => toggleBandFilter(RISK_BAND_ORDER[legendItem.index]),
        }
      : { display: false };
    state.charts[chartKey] = new Chart(canvas, {
      type: "doughnut",
      data: {
        labels: RISK_BAND_ORDER,
        datasets: [{ data: counts, backgroundColor: RISK_BAND_ORDER.map((b) => CHART_COLORS[b]) }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: legendConfig },
        onClick: interactive
          ? (evt, elements) => {
              if (!elements.length) return;
              toggleBandFilter(RISK_BAND_ORDER[elements[0].index]);
            }
          : undefined,
      },
    });
  }

  function renderByProductChart(rows) {
    const canvas = el("chart-by-product");
    if (!canvas) return;
    destroyChart("byProduct");
    const products = [...new Set(rows.map((t) => t.product_cd))].sort();
    const counts = products.map((p) => rows.filter((t) => t.product_cd === p && (t.risk_band === "High" || t.risk_band === "Critical")).length);
    // Two distinct empty cases, worth telling apart rather than both
    // rendering as the same blank chart: no rows match the filters at
    // all, vs. rows exist but none happen to be High/Critical (a real,
    // informative result -- not a bug -- that an all-zero bar chart would
    // otherwise show as a confusing flat line with no explanation).
    const emptyMsg = el("chart-by-product-empty");
    if (!rows.length) {
      if (emptyMsg) emptyMsg.textContent = "No synthetic transactions match the current filters.";
      toggleChartEmptyState("chart-by-product", true);
      return;
    }
    if (!counts.some((c) => c > 0)) {
      if (emptyMsg) emptyMsg.textContent = "No High or Critical transactions in the current scope.";
      toggleChartEmptyState("chart-by-product", true);
      return;
    }
    if (emptyMsg) emptyMsg.textContent = "No synthetic transactions match the current filters.";
    toggleChartEmptyState("chart-by-product", false);
    state.charts.byProduct = new Chart(canvas, {
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

  // Both the scatter and sequence charts below plot risk_score on the
  // y-axis. A fixed 0-1 range is honest but, since scores in this
  // project's synthetic batches typically cluster narrowly (documented
  // in docs/DASHBOARD.md), it visually wastes most of the chart's
  // height -- the exact same problem the risk-score distribution
  // histogram already solved by auto-ranging to the batch's own observed
  // range instead of a fixed axis. Reused here for the same honest
  // reason: the shape stays readable without implying a spread that
  // isn't there.
  function scoreAxisRange(rows) {
    if (!rows.length) return { min: 0, max: 1 };
    const scores = rows.map((t) => t.risk_score);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const pad = (max - min) * 0.1 || 0.02;
    return { min: Math.max(0, min - pad), max: Math.min(1, max + pad) };
  }

  function renderAmountVsRiskChart(rows) {
    const canvas = el("chart-amount-vs-risk");
    if (!canvas) return;
    destroyChart("amountVsRisk");
    toggleChartEmptyState("chart-amount-vs-risk", !rows.length);
    const yRange = scoreAxisRange(rows);
    const noteEl = el("amount-risk-note");
    if (noteEl) {
      noteEl.textContent = rows.length
        ? "Y-axis zoomed to this batch's observed score range (" + yRange.min.toFixed(3) + "–" + yRange.max.toFixed(3) + "), not a fixed 0–1 axis, for the same reason as the distribution chart."
        : "No transactions match the current filters.";
    }
    if (!rows.length) return;
    state.charts.amountVsRisk = new Chart(canvas, {
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
          y: {
            title: { display: true, text: "Risk score", color: "#a9adb8" },
            min: yRange.min,
            max: yRange.max,
            ticks: { color: "#a9adb8" },
            grid: { color: "#30343d" },
          },
        },
      }),
    });
  }

  function renderSequenceChart(rows) {
    const canvas = el("chart-sequence");
    if (!canvas) return;
    destroyChart("sequence");
    toggleChartEmptyState("chart-sequence", !rows.length);
    if (!rows.length) return;
    const bySeq = [...rows].sort((a, b) => a.sequence - b.sequence);
    const yRange = scoreAxisRange(rows);
    state.charts.sequence = new Chart(canvas, {
      type: "line",
      data: {
        labels: bySeq.map((t) => "#" + t.sequence),
        datasets: [{ label: "Risk score", data: bySeq.map((t) => t.risk_score), borderColor: "#b183f7", backgroundColor: "rgba(177,131,247,0.15)", tension: 0.25, pointRadius: 2 }],
      },
      options: baseOptions({
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#a9adb8", maxTicksLimit: 10 }, grid: { color: "#30343d" } },
          y: { min: yRange.min, max: yRange.max, ticks: { color: "#a9adb8" }, grid: { color: "#30343d" } },
        },
      }),
    });
  }

  // ---------------------------------------------------------------------
  // Detail drawer (shared across every view -- stays open across a
  // navigation, since it's a fixed-position overlay, not part of any
  // one view's content)
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

  let drawerOpener = null;
  function openDrawer(transactionId) {
    if (!state.raw) return;
    const t = state.raw.transactions.find((x) => x.transaction_id === transactionId);
    if (!t) return;

    if (!el("detail-drawer").classList.contains("open")) drawerOpener = document.activeElement;
    state.selectedId = transactionId;
    window.CaseUI.show(transactionId);
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
    el("drawer-close").focus();
  }

  function closeDrawer() {
    const wasOpen = el("detail-drawer").classList.contains("open");
    window.CaseUI.close();
    el("detail-drawer").classList.remove("open");
    el("detail-drawer").setAttribute("aria-hidden", "true");
    el("drawer-backdrop").classList.remove("open");
    if (wasOpen) {
      const replacement = [...document.querySelectorAll(".view.active button[data-id]")].find(x => x.dataset.id === state.selectedId);
      (drawerOpener?.isConnected ? drawerOpener : replacement || document.querySelector(".view.active .view-heading"))?.focus();
    }
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
      ["case-status-filter","case-analyst-filter","case-priority-filter"].forEach(id => el(id).value = "");
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
      if (e.key === "Tab" && el("detail-drawer").classList.contains("open")) {
        const focusable = [...el("detail-drawer").querySelectorAll('button,select,textarea,a[href],summary')].filter(x => !x.disabled && x.getClientRects().length);
        const first = focusable[0], last = focusable.at(-1);
        if (!el("detail-drawer").contains(document.activeElement)) {e.preventDefault();first.focus();}
        else if (e.shiftKey && document.activeElement === first) {e.preventDefault();last.focus();}
        else if (!e.shiftKey && document.activeElement === last) {e.preventDefault();first.focus();}
      }
      if (e.key !== "Escape") return;
      closeDrawer();
      closeNav();
    });

    el("kanban-board").addEventListener("scroll", updateKanbanScrollAffordance, { passive: true });
    window.addEventListener("resize", updateKanbanScrollAffordance);

    el("nav-toggle").addEventListener("click", openNav);
    el("sidenav-close").addEventListener("click", closeNav);
    el("nav-backdrop").addEventListener("click", closeNav);
    // Route links close the mobile nav on click; renderActiveView() (the
    // hashchange listener) also calls closeNav() unconditionally, so this
    // covers back/forward navigation too, not just direct link clicks.
    document.querySelectorAll(".sidenav-list a[href^='#']").forEach((link) => {
      link.addEventListener("click", closeNav);
    });

    window.addEventListener("hashchange", renderActiveView);
  }

  async function init() {
    wireControls();
    renderActiveView(); // show the correct view immediately, even before data loads
    await checkHealth();
    await loadQueue(state.seed);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
