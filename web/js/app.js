import { api, ApiError } from "./api.js";
import { debounce, fmt, h, img, persist, store } from "./dom.js";

const EXAMPLES = [
  "Something like Attack on Titan but funnier, under 25 episodes",
  "A cozy slice of life with no romance, something recent",
  "I loved Steins;Gate and Monster but hated Sword Art Online",
  "A classic mecha show with a great story",
  "Dark fantasy with a morally grey hero",
];
const MODE_LABEL = { search: "🔎 Search", similar: "🧭 Similar to", taste: "🎯 Taste profile" };
const FILTER_LABEL = {
  min_score: "score ≥", year_min: "from", year_max: "until", max_episodes: "≤ episodes",
  types: "format", include_tags: "with", exclude_tags: "without",
};
const emptyFilters = () => ({
  min_score: null, year_min: null, year_max: null, max_episodes: null,
  types: [], include_tags: [], exclude_tags: [],
});

const state = {
  liked: store("anirec.liked", []),
  disliked: store("anirec.disliked", []),
  filters: { ...emptyFilters(), ...store("anirec.filters", {}) },
  explain: store("anirec.explain", true),
  limit: store("anirec.limit", 6),
  facets: null,
  busy: false,
};

const $ = (id) => document.getElementById(id);
const thread = $("thread");

// ---------------------------------------------------------------- helpers

function toast(message, kind = "") {
  const el = h("div", { class: `toast ${kind}`, text: message });
  $("toasts").append(el);
  setTimeout(() => el.remove(), 3500);
}

function shortModel(name = "") {
  return name.split("/").pop().replace("embeddinggemma-300m", "EmbeddingGemma")
    .replace(/^gemini-/, "Gemini ").replace(/-flash/, " Flash").replace(/-lite/, " Lite");
}

function ref(anime) {
  return { anime_id: anime.anime_id, title: anime.title, title_english: anime.title_english ?? null, image_url: anime.image_url ?? null };
}

function filtersPayload() {
  const f = state.filters;
  const num = (v) => (v === "" || v === null || v === undefined || Number.isNaN(Number(v)) ? null : Number(v));
  return {
    min_score: num(f.min_score) || null,
    year_min: num(f.year_min),
    year_max: num(f.year_max),
    max_episodes: num(f.max_episodes),
    types: f.types,
    include_tags: f.include_tags,
    exclude_tags: f.exclude_tags,
  };
}

function activeFilterCount() {
  const f = filtersPayload();
  return ["min_score", "year_min", "year_max", "max_episodes"].filter((k) => f[k] !== null).length
    + f.types.length + f.include_tags.length + f.exclude_tags.length;
}

function setCount(el, n) {
  el.hidden = n === 0;
  el.textContent = String(n);
}

function scrollToEnd() {
  requestAnimationFrame(() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }));
}

// ---------------------------------------------------------------- taste

function isLiked(id) { return state.liked.some((a) => a.anime_id === id); }
function isDisliked(id) { return state.disliked.some((a) => a.anime_id === id); }

function setTaste(anime, list) {
  const id = anime.anime_id;
  state.liked = state.liked.filter((a) => a.anime_id !== id);
  state.disliked = state.disliked.filter((a) => a.anime_id !== id);
  if (list) state[list].push(ref(anime));
  persist("anirec.liked", state.liked);
  persist("anirec.disliked", state.disliked);
  renderTaste();
}

function renderTaste() {
  setCount($("taste-count"), state.liked.length + state.disliked.length);
  for (const [list, el] of [["liked", $("liked-list")], ["disliked", $("disliked-list")]]) {
    el.replaceChildren(...state[list].map((a) =>
      h("li", { class: "pick" },
        img(a.image_url, ""),
        h("div", { class: "s-text" },
          h("div", { class: "s-title", text: a.title }),
          a.title_english && a.title_english !== a.title ? h("div", { class: "s-sub", text: a.title_english }) : null),
        h("button", { class: "icon-btn", type: "button", "aria-label": `Remove ${a.title}`, text: "✕", onclick: () => setTaste(a, null) }),
      )));
  }
}

const searchTitles = debounce(async (q) => {
  const list = $("suggestions");
  if (!q.trim()) return list.replaceChildren();
  try {
    const results = await api.titles(q, 8);
    list.replaceChildren(...results.map((a) =>
      h("li", { class: "suggestion", role: "option" },
        img(a.image_url, ""),
        h("div", { class: "s-text" },
          h("div", { class: "s-title", text: a.title }),
          h("div", { class: "s-sub", text: [a.title_english !== a.title ? a.title_english : null, a.members ? `${fmt.format(a.members)} members` : null].filter(Boolean).join(" · ") })),
        h("button", { class: `btn small ${isLiked(a.anime_id) ? "on-good" : ""}`, type: "button", text: "👍", title: "Liked", onclick: () => { setTaste(a, "liked"); list.replaceChildren(); $("title-search").value = ""; } }),
        h("button", { class: "btn small", type: "button", text: "👎", title: "Disliked", onclick: () => { setTaste(a, "disliked"); list.replaceChildren(); $("title-search").value = ""; } }),
      )));
  } catch (err) {
    toast(err.message, "err");
  }
}, 180);

// ---------------------------------------------------------------- filters

function renderFilters() {
  const f = state.filters;
  const facets = state.facets;
  $("min-score").value = f.min_score || 0;
  $("min-score-out").textContent = f.min_score ? `${Number(f.min_score).toFixed(1)}+` : "any";
  $("year-min").value = f.year_min ?? "";
  $("year-max").value = f.year_max ?? "";
  $("max-episodes").value = f.max_episodes ?? "";
  if (facets) {
    $("year-min").placeholder = facets.year_min ?? "from";
    $("year-max").placeholder = facets.year_max ?? "to";
    $("type-chips").replaceChildren(...facets.type.map(({ value, count }) =>
      h("button", {
        class: `toggle ${f.types.includes(value) ? "on" : ""}`, type: "button", "aria-pressed": String(f.types.includes(value)),
        onclick: () => { f.types = f.types.includes(value) ? f.types.filter((t) => t !== value) : [...f.types, value]; saveFilters(); },
      }, value, h("small", { text: fmt.format(count) }))));
    const q = $("tag-search").value.trim().toLowerCase();
    const tags = [...facets.genres, ...facets.themes, ...facets.demographics]
      .filter(({ value }) => !q || value.toLowerCase().includes(q));
    $("tag-chips").replaceChildren(...tags.map(({ value, count }) => {
      const mode = f.include_tags.includes(value) ? "include" : f.exclude_tags.includes(value) ? "exclude" : "";
      return h("button", {
        class: `toggle ${mode}`, type: "button", title: "include → exclude → off",
        onclick: () => {
          f.include_tags = f.include_tags.filter((t) => t !== value);
          f.exclude_tags = f.exclude_tags.filter((t) => t !== value);
          if (mode === "") f.include_tags.push(value);
          else if (mode === "include") f.exclude_tags.push(value);
          saveFilters();
        },
      }, value, h("small", { text: fmt.format(count) }));
    }));
  }
  setCount($("filters-count"), activeFilterCount());
}

function saveFilters() {
  persist("anirec.filters", state.filters);
  renderFilters();
}

// ---------------------------------------------------------------- drawers & detail

function openDrawer(id) {
  closeDrawers();
  $(id).hidden = false;
  $("scrim").hidden = false;
  $(id).querySelector("input")?.focus();
}

function closeDrawers() {
  for (const d of document.querySelectorAll(".drawer")) d.hidden = true;
  $("scrim").hidden = true;
}

function stat(label, value, cls = "") {
  return value === null || value === undefined || value === "" ? null
    : h("div", { class: `stat ${cls}` }, h("b", { text: value }), h("span", { text: label }));
}

function openDetail(item) {
  const dialog = $("detail");
  const tags = [...item.genres, ...item.themes, ...item.demographics];
  const aired = item.season || (item.start_year ? (item.end_year && item.end_year !== item.start_year ? `${item.start_year}–${item.end_year}` : `${item.start_year}`) : null);
  const names = [item.title_english, item.title_japanese].filter((n) => n && n !== item.title);
  const people = [
    item.studios.length ? `Studio ${item.studios.join(", ")}` : null,
    item.directors.length ? `Directed by ${item.directors.join(", ")}` : null,
    item.original_creators.length ? `Original work by ${item.original_creators.join(", ")}` : null,
  ].filter(Boolean);

  dialog.replaceChildren(
    h("div", { class: "detail-scroll" },
      h("button", { class: "icon-btn detail-close", type: "button", "aria-label": "Close", text: "✕", onclick: () => dialog.close() }),
      h("div", { class: "detail-hero" },
        h("div", { class: "detail-backdrop", style: `background-image:url("${encodeURI(item.image_url)}")` }),
        img(item.image_url, item.title, "detail-poster"),
        h("div", {},
          h("h2", { text: item.title }),
          names.length ? h("p", { class: "alt", text: names.join(" · ") }) : null,
          h("div", { class: "chips", style: "margin-top:12px" }, tags.map((t) => h("span", { class: "chip", text: t }))),
          h("div", { class: "stats" },
            stat("MAL score", item.score.toFixed(2), "gold"),
            stat("Rank", `#${fmt.format(item.rank)}`),
            stat("Members", fmt.format(item.members)),
            stat("Format", item.type),
            stat("Episodes", item.episodes ?? (item.is_ongoing ? "Ongoing" : "?")),
            stat("Aired", aired),
            stat("Source", item.source),
            stat("Rating", item.age_rating?.split(" - ")[0]),
          ),
          item.reason ? h("p", { class: "callout", text: `💡 ${item.reason}` }) : null,
          h("div", { class: "detail-actions" },
            h("button", { class: "btn primary", type: "button", text: "More like this", onclick: () => { dialog.close(); askSimilar(item); } }),
            h("button", { class: `btn ${isLiked(item.anime_id) ? "on-good" : ""}`, type: "button", text: isLiked(item.anime_id) ? "✓ Liked" : "👍 Like", onclick: (e) => { setTaste(item, isLiked(item.anime_id) ? null : "liked"); e.target.textContent = isLiked(item.anime_id) ? "✓ Liked" : "👍 Like"; e.target.classList.toggle("on-good"); } }),
            h("button", { class: "btn", type: "button", text: "👎 Not for me", onclick: () => { setTaste(item, "disliked"); toast(`Will steer away from ${item.title}`); } }),
            h("a", { class: "btn ghost", href: item.mal_url, target: "_blank", rel: "noopener", text: "MyAnimeList ↗" }),
          ),
        ),
      ),
      h("div", { class: "detail-body" },
        h("section", {}, h("h3", { text: "Synopsis" }), h("p", { class: "synopsis", text: item.synopsis || "No synopsis available." })),
        people.length ? h("p", { class: "people", text: people.join(" · ") }) : null,
        item.main_characters.length ? h("section", {},
          h("h3", { text: "Characters" }),
          h("div", { class: "characters" }, item.main_characters.map((c) =>
            h("div", { class: "character" },
              img(c.image_url, c.name),
              h("b", { text: c.name }),
              h("span", { text: [c.role, c.voice_actor ? `CV ${c.voice_actor}` : null].filter(Boolean).join(" · ") }))))) : null,
      ),
    ),
  );
  dialog.showModal();
  dialog.querySelector(".detail-scroll").scrollTop = 0;
}

// ---------------------------------------------------------------- rendering

function card(item, index, explained) {
  const facts = [item.type, item.episodes ? `${item.episodes} ep` : item.is_ongoing ? "ongoing" : null, item.start_year].filter(Boolean).join(" · ");
  const tags = [...item.genres, ...item.themes].slice(0, 3);
  const el = h("article", {
    class: "card", tabindex: "0", "aria-label": item.title,
    onclick: (e) => { if (!e.target.closest("button")) openDetail(item); },
    onkeydown: (e) => { if (e.key === "Enter" && e.target === el) openDetail(item); },
  },
    h("div", { class: "poster" },
      img(item.image_url, item.title),
      h("span", { class: "badge score", text: `★ ${item.score.toFixed(2)}` }),
      explained && index === 0 ? h("span", { class: "badge pick", text: "Top pick" }) : null,
      h("div", { class: "poster-title", text: item.title }),
    ),
    h("div", { class: "card-body" },
      h("p", { class: "facts", text: facts }),
      tags.length ? h("div", { class: "chips" }, tags.map((t) => h("span", { class: "chip", text: t }))) : null,
      item.reason ? h("p", { class: "reason", text: item.reason }) : null,
      h("div", { class: "card-actions" },
        h("button", { class: "btn small", type: "button", text: "More like this", onclick: () => askSimilar(item) }),
        h("button", {
          class: `btn small like ${isLiked(item.anime_id) ? "on-good" : ""}`, type: "button", "aria-label": `Like ${item.title}`,
          text: isLiked(item.anime_id) ? "✓" : "👍",
          onclick: (e) => {
            const liked = !isLiked(item.anime_id);
            setTaste(item, liked ? "liked" : null);
            e.target.textContent = liked ? "✓" : "👍";
            e.target.classList.toggle("on-good", liked);
            if (liked) toast(`Added ${item.title} to your taste`);
          },
        }),
      ),
    ),
  );
  return el;
}

function understoodChips(u, seeds) {
  const chips = [];
  if (!u) {
    if (seeds?.[0]) chips.push(h("span", { class: "u-chip mode", text: MODE_LABEL.similar }), refChip(seeds[0], "good"));
    return chips;
  }
  chips.push(h("span", { class: "u-chip mode", text: MODE_LABEL[u.mode] }));
  u.liked.forEach((a) => chips.push(refChip(a, "good")));
  u.disliked.forEach((a) => chips.push(refChip(a, "bad", "not ")));
  if (u.query) chips.push(h("span", { class: "u-chip plain", text: `“${u.query}”` }));
  for (const [key, value] of Object.entries(u.filters)) {
    if (value === null || value === false || (Array.isArray(value) && !value.length)) continue;
    chips.push(h("span", { class: "u-chip plain", text: `${FILTER_LABEL[key] || key} ${Array.isArray(value) ? value.join(", ") : value}` }));
  }
  if (u.unresolved_titles.length) chips.push(h("span", { class: "u-chip warn", text: `⚠ not found: ${u.unresolved_titles.join(", ")}` }));
  return chips;
}

function refChip(a, cls, prefix = "") {
  return h("span", { class: `u-chip ${cls}` }, img(a.image_url, ""), `${prefix}${a.title}`);
}

function renderAnswer(container, response) {
  const items = response.items;
  container.replaceChildren(
    h("div", { class: "understood" }, understoodChips(response.understood, response.seeds)),
    response.summary ? h("p", { class: "summary", text: response.summary }) : null,
    items.length
      ? h("div", { class: "grid" }, items.map((item, i) => card(item, i, response.explained)))
      : h("div", { class: "error-box", text: "Nothing matched. Try loosening the filters or rephrasing." }),
    h("p", { class: "meta", text: `${fmt.format(response.took_ms)} ms · ${response.explained ? "re-ranked and explained by Gemini" : "vector search"}` }),
  );
}

function skeleton(n) {
  return h("div", {},
    h("div", { class: "typing", "aria-label": "Thinking" }, h("span"), h("span"), h("span")),
    h("div", { class: "grid" }, Array.from({ length: n }, () =>
      h("div", { class: "card skeleton" }, h("div", { class: "poster" }),
        h("div", { class: "card-body" }, h("div", { class: "line" }), h("div", { class: "line short" }))))));
}

// ---------------------------------------------------------------- conversation

async function run(userText, request) {
  if (state.busy) return;
  state.busy = true;
  $("composer").querySelector(".send").disabled = true;
  document.body.classList.add("has-thread");

  const user = h("div", { class: "turn-user" }, h("div", { class: "bubble", text: userText }));
  thread.append(user);
  const bot = h("div", { class: "turn-bot" }, skeleton(state.limit));
  thread.append(bot);
  scrollToEnd();

  try {
    renderAnswer(bot, await request());
  } catch (err) {
    bot.replaceChildren(h("div", { class: "error-box", text: err instanceof ApiError ? err.message : "Something went wrong." }));
  } finally {
    state.busy = false;
    $("composer").querySelector(".send").disabled = false;
    // Show the question with its answer; scroll-margin keeps it clear of the sticky top bar.
    user.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function ask(message) {
  const text = message.trim();
  if (!text) return;
  run(text, () => api.chat({
    message: text,
    filters: filtersPayload(),
    liked: state.liked.map((a) => a.anime_id),
    disliked: state.disliked.map((a) => a.anime_id),
    limit: Number(state.limit),
    explain: state.explain,
  }));
}

function askSimilar(item) {
  run(`More like ${item.title}`, () => api.similar({
    anime_id: item.anime_id, filters: filtersPayload(), limit: Number(state.limit), explain: state.explain,
  }));
}

// ---------------------------------------------------------------- boot

function wire() {
  const input = $("message");
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    ask(input.value);
    input.value = "";
    input.style.height = "";
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("composer").requestSubmit();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  });

  $("explain").checked = state.explain;
  $("explain").addEventListener("change", (e) => { state.explain = e.target.checked; persist("anirec.explain", state.explain); });
  $("limit").value = String(state.limit);
  $("limit").addEventListener("change", (e) => { state.limit = Number(e.target.value); persist("anirec.limit", state.limit); });

  $("taste-btn").addEventListener("click", () => openDrawer("taste-drawer"));
  $("filters-btn").addEventListener("click", () => openDrawer("filters-drawer"));
  $("scrim").addEventListener("click", closeDrawers);
  document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closeDrawers));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawers(); });
  $("detail").addEventListener("click", (e) => { if (e.target === $("detail")) $("detail").close(); });
  $("new-chat").addEventListener("click", () => {
    thread.replaceChildren();
    document.body.classList.remove("has-thread");
    window.scrollTo({ top: 0 });
    input.focus();
  });

  $("title-search").addEventListener("input", (e) => searchTitles(e.target.value));
  $("tag-search").addEventListener("input", renderFilters);
  $("min-score").addEventListener("input", (e) => { state.filters.min_score = Number(e.target.value) || null; saveFilters(); });
  for (const [id, key] of [["year-min", "year_min"], ["year-max", "year_max"], ["max-episodes", "max_episodes"]]) {
    $(id).addEventListener("change", (e) => { state.filters[key] = e.target.value === "" ? null : Number(e.target.value); saveFilters(); });
  }
  $("reset-filters").addEventListener("click", () => { state.filters = emptyFilters(); $("tag-search").value = ""; saveFilters(); });

  $("examples").replaceChildren(...EXAMPLES.map((text) =>
    h("button", { class: "example", type: "button", text, onclick: () => ask(text) })));
}

async function boot() {
  wire();
  renderTaste();
  renderFilters();

  const status = $("status");
  try {
    const ready = await api.ready();
    status.classList.add("ok");
    $("status-text").textContent = `${fmt.format(ready.points)} anime · ${shortModel(ready.embedding_model)} · ${ready.chat_models.length ? shortModel(ready.chat_models[0]) : "no LLM"}`;
    $("hero-eyebrow").textContent = `${fmt.format(ready.points)} anime · semantic search · grounded AI picks`;
  } catch (err) {
    status.classList.add("err");
    $("status-text").textContent = "API offline";
    toast(err.message, "err");
    return;
  }

  const [facets, popular] = await Promise.allSettled([api.facets(), api.popular(48)]);
  if (facets.status === "fulfilled") {
    state.facets = facets.value;
    renderFilters();
  }
  if (popular.status === "fulfilled") {
    $("poster-wall").replaceChildren(...popular.value.map((a) => img(a.image_url, "")));
  }
}

boot();
