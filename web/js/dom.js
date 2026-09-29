// Tiny safe DOM builder. Text always goes through textContent, never innerHTML, so data
// from the dataset or the LLM (synopses, reasons, titles) can never inject markup.

export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key.startsWith("on")) el.addEventListener(key.slice(2).toLowerCase(), value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function img(src, alt, className) {
  if (!src) return h("div", { class: `${className || ""} noimg`, "aria-hidden": "true", text: "?" });
  const el = h("img", { src, alt: alt || "", loading: "lazy", decoding: "async", class: className, referrerpolicy: "no-referrer" });
  el.addEventListener("error", () => el.classList.add("img-error"), { once: true });
  return el;
}

export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

export const fmt = new Intl.NumberFormat("en");

export function store(key, fallback) {
  // localStorage can be blocked (private mode); the app must work without it.
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

export function persist(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}
