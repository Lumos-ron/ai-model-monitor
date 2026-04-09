/* AI Model Monitor — frontend renderer */

const DATA_URL = "data/models.json";

const LEADERBOARD_LABELS = {
  lmarena_elo: "LMArena",
  livebench_avg: "LiveBench",
  aa_intelligence_index: "AA Index",
};

const STATUS_LABEL = {
  ok:    { zh: "最新",       en: "Fresh" },
  stale: { zh: "暂无数据",   en: "No data yet" },
  error: { zh: "更新失败",   en: "Update failed" },
};

const I18N_NUM = {
  zh: { noScores: "暂无公布得分", noThird: "暂无榜单数据" },
  en: { noScores: "No published scores", noThird: "No leaderboard data" },
};

/* ---------- theme + language ---------- */

function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.dataset.theme = saved;
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const now = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = now;
    localStorage.setItem("theme", now);
  });
}

const LANG_CYCLE = ["zh", "en", "both"];

function initLang() {
  const saved = localStorage.getItem("lang") || "zh";
  document.documentElement.dataset.lang = saved;
  document.documentElement.lang = saved === "en" ? "en" : "zh";
  document.getElementById("lang-toggle").addEventListener("click", () => {
    const curr = document.documentElement.dataset.lang || "zh";
    const next = LANG_CYCLE[(LANG_CYCLE.indexOf(curr) + 1) % LANG_CYCLE.length];
    document.documentElement.dataset.lang = next;
    document.documentElement.lang = next === "en" ? "en" : "zh";
    localStorage.setItem("lang", next);
    // Re-render so dynamic strings (like status labels) pick up the change
    if (window.__lastData) render(window.__lastData);
  });
}

function currentLang() {
  const v = document.documentElement.dataset.lang || "zh";
  return v === "en" ? "en" : "zh"; // "both" falls back to zh for dynamic strings
}

/* ---------- fetch + render ---------- */

async function loadData() {
  const res = await fetch(`${DATA_URL}?t=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${DATA_URL}: ${res.status}`);
  return res.json();
}

function formatUpdated(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(currentLang() === "en" ? "en-US" : "zh-CN", {
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit",
  });
}

function clampPercent(score, unit) {
  // Most benchmarks are percentages; clamp to 0–100 for the bar width.
  if (unit === "%" || unit === "" || unit == null) {
    return Math.max(0, Math.min(100, Number(score) || 0));
  }
  // Non-percent (e.g. raw Elo) → no bar
  return null;
}

function formatScoreValue(score, unit) {
  if (typeof score !== "number") return String(score ?? "");
  const rounded = Number.isInteger(score) ? score : Math.round(score * 10) / 10;
  if (unit === "%") return `${rounded}%`;
  if (unit) return `${rounded} ${unit}`;
  return `${rounded}`;
}

function render(data) {
  window.__lastData = data;

  document.getElementById("last-updated").textContent =
    (currentLang() === "en" ? "Updated " : "更新于 ") + formatUpdated(data.last_updated);

  const grid = document.getElementById("vendor-grid");
  grid.innerHTML = "";
  const tpl = document.getElementById("card-template");
  const lang = currentLang();

  for (const v of data.vendors) {
    const node = tpl.content.firstElementChild.cloneNode(true);

    node.querySelector(".vendor-name").textContent = lang === "en" ? v.name_en : v.name_zh;
    node.querySelector(".product").textContent = v.product || "";

    const statusEl = node.querySelector(".status");
    const statusKey = v.fetch_status in STATUS_LABEL ? v.fetch_status : "stale";
    statusEl.textContent = STATUS_LABEL[statusKey][lang];
    statusEl.classList.add(statusKey);
    if (v.fetch_error) statusEl.title = v.fetch_error;

    const lm = v.latest_model || {};
    node.querySelector(".model-name").textContent = lm.display_name || "—";
    node.querySelector(".release-date").textContent = lm.release_date || "";

    const srcLink = node.querySelector(".source");
    if (lm.source_url) {
      srcLink.href = lm.source_url;
      srcLink.textContent = lang === "en" ? "Source ↗" : "来源 ↗";
    } else {
      srcLink.remove();
    }

    // Official scores with progress bars
    const ul = node.querySelector(".official-scores");
    const scores = v.official_scores || [];
    if (scores.length === 0) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = I18N_NUM[lang].noScores;
      ul.appendChild(li);
    } else {
      for (const s of scores) {
        const li = document.createElement("li");
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = s.benchmark;
        const val = document.createElement("span");
        val.className = "value";
        val.textContent = formatScoreValue(s.score, s.unit || "%");
        const bar = document.createElement("div");
        bar.className = "bar";
        const pct = clampPercent(s.score, s.unit);
        if (pct !== null) {
          const fill = document.createElement("div");
          fill.className = "bar-fill";
          fill.style.width = `${pct}%`;
          bar.appendChild(fill);
        }
        li.append(name, val, bar);
        ul.appendChild(li);
      }
    }

    // Third-party leaderboard badges
    const tpUl = node.querySelector(".third-party-scores");
    const tp = v.third_party_scores || {};
    const tpKeys = Object.keys(tp).filter((k) => tp[k] != null && tp[k] !== 0);
    if (tpKeys.length === 0) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = I18N_NUM[lang].noThird;
      tpUl.appendChild(li);
    } else {
      for (const k of tpKeys) {
        const li = document.createElement("li");
        const label = document.createElement("span");
        label.textContent = LEADERBOARD_LABELS[k] || k;
        const num = document.createElement("span");
        num.className = "num";
        const v2 = tp[k];
        num.textContent = Number.isInteger(v2) ? v2 : Math.round(v2 * 10) / 10;
        li.append(label, num);
        tpUl.appendChild(li);
      }
    }

    grid.appendChild(node);
  }
}

/* ---------- boot ---------- */

(async function boot() {
  initTheme();
  initLang();
  try {
    const data = await loadData();
    render(data);
  } catch (err) {
    console.error(err);
    document.getElementById("vendor-grid").innerHTML =
      `<p class="empty">${err.message}</p>`;
  }
})();
