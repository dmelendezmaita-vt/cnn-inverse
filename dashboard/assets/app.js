const DATA_ROOT = "./data";

function formatInt(value) {
  return Number(value).toLocaleString();
}

function formatGib(value) {
  return `${Number(value).toFixed(2)} GiB`;
}

function formatValue(value, unit) {
  if (unit === "%") return `${Number(value).toFixed(2)}%`;
  if (unit === "p") return Number(value).toFixed(3);
  if (unit === "runs/hour") return Number(value).toFixed(2);
  if (unit === "seconds") return `${Number(value).toFixed(2)} s`;
  if (unit === "MAE") return Number(value).toFixed(2);
  if (unit === "mean abs log10 error") return Number(value).toFixed(5);
  return Number(value).toFixed(3);
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

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function renderInventory(summary) {
  const kpiContainer = document.getElementById("inventory-kpis");
  kpiContainer.innerHTML = summary.inventory.kpis.map((item) => `
    <article class="kpi-card">
      <h3>${item.label}</h3>
      <p class="kpi-value">${item.value}</p>
      <p class="kpi-note">${item.note}</p>
    </article>
  `).join("");

  const packageBody = document.querySelector("#inventory-table tbody");
  packageBody.innerHTML = summary.inventory.packages.map((item) => `
    <tr>
      <td>${item.label}</td>
      <td>${formatInt(item.files)}</td>
      <td>${formatGib(item.gib)}</td>
      <td>${item.role}</td>
    </tr>
  `).join("");

  const surfaceBody = document.querySelector("#surface-table tbody");
  surfaceBody.innerHTML = summary.inventory.rows_by_space.map((item) => `
    <tr>
      <td>${item.space}</td>
      <td>${formatInt(item.rows)}</td>
      <td>${formatInt(item.numeric_cells)}</td>
    </tr>
  `).join("");

  setText("inventory-note", summary.inventory.interpretation_note);
}

function renderReadingGuide(summary) {
  const container = document.getElementById("reading-guide");
  container.innerHTML = summary.reading_guide.map((item) => {
    const pillClass = item.kind === "boundary" ? "warn" : item.kind === "support" ? "success" : "";
    return `
      <article class="text-card">
        <p><span class="pill ${pillClass}">${item.badge}</span></p>
        <h3>${item.title}</h3>
        <p>${item.body}</p>
      </article>
    `;
  }).join("");
}

function renderLegend(series) {
  const list = document.createElement("ul");
  list.className = "chart-legend";
  series.forEach((item) => {
    const entry = document.createElement("li");
    entry.innerHTML = `<span class="legend-swatch" style="background:${item.color}"></span>${item.label}`;
    list.appendChild(entry);
  });
  return list;
}

function renderBarChart(chart) {
  const width = 680;
  const left = 190;
  const right = 90;
  const top = 12;
  const bottom = 26;
  const groupHeight = 54;
  const innerWidth = width - left - right;
  const seriesCount = chart.series.length;
  const barGap = 6;
  const barHeight = Math.max(12, (groupHeight - barGap * (seriesCount - 1)) / seriesCount);
  const height = top + bottom + chart.categories.length * groupHeight;

  const values = chart.series.flatMap((s) => s.values);
  const minValue = Math.min(0, ...values);
  const maxValue = Math.max(0, ...values);
  const span = maxValue - minValue || 1;
  const scaleX = (value) => left + ((value - minValue) / span) * innerWidth;
  const zeroX = scaleX(0);

  const wrapper = document.createElement("div");
  wrapper.className = "chart-shell";

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": chart.title
  });

  [minValue, 0, maxValue].forEach((tick) => {
    const x = scaleX(tick);
    svg.appendChild(svgEl("line", {
      x1: x,
      y1: top,
      x2: x,
      y2: height - bottom,
      class: tick === 0 ? "zero-line" : "grid-line"
    }));
    const label = svgEl("text", {
      x,
      y: height - 6,
      "text-anchor": "middle",
      class: "axis-label"
    });
    label.textContent = formatValue(tick, chart.unit);
    svg.appendChild(label);
  });

  chart.categories.forEach((category, categoryIndex) => {
    const groupTop = top + categoryIndex * groupHeight;
    const label = svgEl("text", {
      x: left - 12,
      y: groupTop + groupHeight / 2 + 4,
      "text-anchor": "end",
      class: "axis-label"
    });
    label.textContent = category;
    svg.appendChild(label);

    chart.series.forEach((series, seriesIndex) => {
      const value = series.values[categoryIndex];
      const y = groupTop + seriesIndex * (barHeight + barGap);
      const x = Math.min(zeroX, scaleX(value));
      const w = Math.abs(scaleX(value) - zeroX);
      svg.appendChild(svgEl("rect", {
        x,
        y,
        width: Math.max(w, 1),
        height: barHeight,
        rx: 4,
        fill: series.color
      }));
      const valueLabel = svgEl("text", {
        x: scaleX(value) + (value >= 0 ? 6 : -6),
        y: y + barHeight / 2 + 4,
        "text-anchor": value >= 0 ? "start" : "end",
        class: "value-label"
      });
      valueLabel.textContent = formatValue(value, chart.unit);
      svg.appendChild(valueLabel);
    });
  });

  wrapper.appendChild(renderLegend(chart.series));
  wrapper.appendChild(svg);
  return wrapper;
}

function renderThreadCard(card) {
  return `
    <article class="text-card">
      <h3>${card.title}</h3>
      <p>${card.body}</p>
    </article>
  `;
}

function renderThreadSection(thread) {
  const section = document.createElement("section");
  section.className = "panel thread-section";

  const pillClass = thread.badge === "Primary evidence"
    ? ""
    : thread.badge === "Supporting evidence"
      ? "success"
      : thread.badge === "Coverage expansion"
        ? "purple"
        : "warn";

  section.innerHTML = `
    <div class="thread-header">
      <p><span class="pill ${pillClass}">${thread.badge}</span></p>
      <h2>${thread.title}</h2>
      <p>${thread.intro}</p>
      <p class="panel-note"><strong>How to read this section:</strong> ${thread.how_to_read}</p>
      <p class="panel-note"><strong>Main conclusion:</strong> ${thread.main_conclusion}</p>
    </div>
  `;

  const chartGrid = document.createElement("div");
  chartGrid.className = "chart-grid";

  thread.charts.forEach((chart) => {
    const card = document.createElement("article");
    card.className = "chart-card";
    card.innerHTML = `
      <h3>${chart.title}</h3>
      <p class="chart-subtitle">${chart.subtitle}</p>
    `;
    card.appendChild(renderBarChart(chart));
    const notes = document.createElement("ul");
    notes.className = "chart-notes";
    notes.innerHTML = chart.notes.map((note) => `<li>${note}</li>`).join("");
    card.appendChild(notes);
    chartGrid.appendChild(card);
  });

  section.appendChild(chartGrid);

  if (thread.cards && thread.cards.length) {
    const cards = document.createElement("div");
    cards.className = "cards-grid";
    cards.innerHTML = thread.cards.map(renderThreadCard).join("");
    section.appendChild(cards);
  }

  return section;
}

async function main() {
  const summary = await loadJson(`${DATA_ROOT}/canonical_results.json`);
  setText("generated-on", `Updated ${summary.generated_on}. ${summary.scope_note}`);
  renderInventory(summary);
  renderReadingGuide(summary);

  const container = document.getElementById("thread-sections");
  summary.threads.forEach((thread) => {
    container.appendChild(renderThreadSection(thread));
  });
}

main().catch((error) => {
  setText("generated-on", `Dashboard load failed: ${error.message}`);
});
