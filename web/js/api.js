// Backend client. nginx proxies /api/* to the FastAPI service, so there is no CORS.

const BASE = "/api";

export class ApiError extends Error {}

async function call(path, { method = "GET", body, signal } = {}) {
  let response;
  try {
    response = await fetch(BASE + path, {
      method,
      signal,
      headers: body ? { "content-type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError("The recommendation API is unreachable.");
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(`${response.status}: ${detail}`);
  }
  return response.json();
}

export const api = {
  ready: () => call("/ready"),
  facets: () => call("/facets"),
  popular: (limit = 40) => call(`/popular?limit=${limit}`),
  titles: (q, limit = 8, signal) => call(`/titles?q=${encodeURIComponent(q)}&limit=${limit}`, { signal }),
  anime: (id) => call(`/anime/${id}`),
  chat: (payload) => call("/chat", { method: "POST", body: payload }),
  similar: (payload) => call("/similar", { method: "POST", body: payload }),
};
