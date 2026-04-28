const DATA_ROOT = "./data";

function formatNum(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function formatPct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toFixed(digits)}%`;
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  return response.json();
}

function renderKpis(summary) {
  const container = document.getElementById("kpi-grid");
  container.innerHTML = summary.kpis.map((item) => `
    <article class="kpi-card">
      <h3>${item.label}</h3>
      <p class="kpi-value">${item.value}</p>
      <p class="kpi-note">${item.note}</p>
    </article>
  `).join("");
}

function renderGuide(summary) {
  const container = document.getElementById("reading-guide");
  container.innerHTML = summary.reading_guide.map((item) => `
    <article class="text-card">
      <h3>${item.title}</h3>
      <p>${item.body}</p>
      <p><span class="pill ${item.kind === "boundary" ? "warn" : ""}">${item.badge}</span></p>
    </article>
  `).join("");
}

function renderPrimary(summary) {
  setText("primary-note", summary.primary_note);
  const body = document.querySelector("#primary-table tbody");
  body.innerHTML = summary.primary_rows.map((row) => `
    <tr>
      <td>${row.class}</td>
      <td>${row.representative}</td>
      <td>${row.meaning}</td>
      <td>${row.mae}</td>
      <td>${row.mse}</td>
      <td>${row.r2}</td>
      <td>${row.resources}</td>
    </tr>
  `).join("");
}

function renderBoundary(summary) {
  const body = document.querySelector("#boundary-table tbody");
  body.innerHTML = summary.boundary_rows.map((row) => `
    <tr>
      <td>${row.study}</td>
      <td>${row.intervention}</td>
      <td>${row.primary_delta}</td>
      <td>${row.secondary_delta}</td>
      <td>${row.resources}</td>
      <td>${row.interpretation}</td>
    </tr>
  `).join("");
}

function renderSupport(summary) {
  const body = document.querySelector("#support-table tbody");
  body.innerHTML = summary.support_rows.map((row) => `
    <tr>
      <td>${row.study}</td>
      <td>${row.question}</td>
      <td>${row.main_result}</td>
      <td>${row.quality_effect}</td>
      <td>${row.runtime_effect}</td>
      <td>${row.interpretation}</td>
    </tr>
  `).join("");
}

function renderBoundaryCards(summary) {
  const container = document.getElementById("data-boundary");
  container.innerHTML = summary.data_boundary.map((item) => `
    <article class="text-card">
      <h3>${item.title}</h3>
      <p>${item.body}</p>
    </article>
  `).join("");
}

async function main() {
  const summary = await loadJson(`${DATA_ROOT}/canonical_results.json`);
  setText("generated-on", `Updated ${summary.generated_on}. ${summary.scope_note}`);
  renderKpis(summary);
  renderGuide(summary);
  renderPrimary(summary);
  renderBoundary(summary);
  renderSupport(summary);
  renderBoundaryCards(summary);
}

main().catch((error) => {
  setText("generated-on", `Dashboard load failed: ${error.message}`);
});
