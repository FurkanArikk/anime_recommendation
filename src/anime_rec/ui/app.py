"""Streamlit chat front end.

Chat in plain language; the sidebar adds hard filters and a liked/disliked taste profile
that every message takes into account. Each answer shows what the assistant understood,
then result cards with poster, scores, tags, the grounded reason, synopsis and characters.
"""

from typing import Any

import streamlit as st

from anime_rec.config import get_settings
from anime_rec.ui import api_client

EXAMPLES = [
    "Something like Attack on Titan but funnier, under 25 episodes",
    "A cozy slice of life with no romance, something recent",
    "I loved Steins;Gate and Monster but hated Sword Art Online",
    "A classic mecha show with a great story",
]
CHARACTERS_SHOWN = 6

st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")


# --- backend ------------------------------------------------------------------------


@st.cache_resource
def get_api() -> api_client.ApiClient:
    return api_client.ApiClient(get_settings().api_url)


@st.cache_data(ttl=3600, show_spinner=False)
def load_catalog() -> tuple[dict[str, Any], dict[str, int]]:
    """Facets for the filter widgets and a label -> anime_id map for the pickers."""
    api = get_api()
    labels: dict[str, int] = {}
    for t in api.titles():
        english = t.get("title_english")
        label = f"{t['title']} ({english})" if english and english != t["title"] else t["title"]
        labels[label] = t["anime_id"]
    return api.facets(), labels


api = get_api()
try:
    facets, title_ids = load_catalog()
    status = api.ready()
except api_client.ApiError as exc:
    st.error(f"The recommendation API is not available yet ({exc}). Is `make up` running?")
    st.stop()

label_of = {anime_id: label for label, anime_id in title_ids.items()}

state = st.session_state
state.setdefault("messages", [])
state.setdefault("liked", [])
state.setdefault("disliked", [])
state.setdefault("pending", None)


# --- sidebar: taste + filters ---------------------------------------------------------

with st.sidebar:
    st.header("Your taste")
    st.multiselect("👍 Liked", options=list(title_ids), key="liked",
                   placeholder="Search titles you enjoyed")  # fmt: skip
    st.multiselect("👎 Disliked", options=list(title_ids), key="disliked",
                   placeholder="Titles to steer away from")  # fmt: skip

    st.header("Filters")
    min_score = st.slider("Minimum MAL score", 0.0, 10.0, 0.0, 0.1)
    y_lo, y_hi = facets.get("year_min") or 1960, facets.get("year_max") or 2025
    years = st.slider("First aired", y_lo, y_hi, (y_lo, y_hi))
    types = st.multiselect("Format", [f["value"] for f in facets["type"]])
    max_episodes = st.number_input("Max episodes (0 = any)", 0, 2000, 0, step=1)
    all_tags = [f["value"] for f in facets["genres"] + facets["themes"] + facets["demographics"]]
    include_tags = st.multiselect("Must include", all_tags, help="Genres, themes, demographics")
    exclude_tags = st.multiselect("Exclude", all_tags)

    st.header("Answer")
    limit = st.slider("Results", 3, 12, 6)
    explain = st.toggle("Explain picks with Gemini", value=True, help="Adds about 3 seconds")
    if st.button("Clear conversation", width="stretch"):
        state.messages = []
    st.caption(f"{status['points']:,} anime · `{status['embedding_model']}` · "
               f"`{status['chat_models'][0] if status['chat_models'] else 'no LLM'}`")  # fmt: skip

filters = {
    "min_score": min_score or None,
    "year_min": years[0] if years[0] != y_lo else None,
    "year_max": years[1] if years[1] != y_hi else None,
    "types": types,
    "max_episodes": int(max_episodes) or None,
    "include_tags": include_tags,
    "exclude_tags": exclude_tags,
}


# --- rendering ------------------------------------------------------------------------


def like(anime_id: int) -> None:
    label = label_of.get(anime_id)
    if label and label not in state.liked:
        state.liked = [*state.liked, label]


def more_like(anime_id: int, title: str) -> None:
    state.pending = {"kind": "similar", "anime_id": anime_id, "title": title}


def render_understood(u: dict[str, Any] | None, seeds: list[dict[str, Any]] | None) -> None:
    if u is None:
        if seeds:
            st.caption(f"🔎 Anime similar to **{seeds[0]['title']}**")
        return
    parts = [
        {"search": "🔎 Searching", "similar": "🧭 Similar to", "taste": "🎯 Taste profile"}[
            u["mode"]
        ]
    ]
    if u["liked"]:
        parts.append("liked: " + ", ".join(f"**{x['title']}**" for x in u["liked"]))
    if u["disliked"]:
        parts.append("avoiding: " + ", ".join(f"**{x['title']}**" for x in u["disliked"]))
    if u["query"]:
        parts.append(f"about: *{u['query']}*")
    active = {k: v for k, v in u["filters"].items() if v not in (None, [], False)}
    if active:
        parts.append("filters: " + ", ".join(f"`{k}={v}`" for k, v in active.items()))
    st.caption(" · ".join(parts))
    if u["unresolved_titles"]:
        st.caption("⚠️ Couldn't find: " + ", ".join(u["unresolved_titles"]))


def render_card(item: dict[str, Any], key: str) -> None:
    with st.container(border=True):
        poster, body = st.columns([1, 4], gap="medium")
        poster.image(item["image_url"], width="stretch")
        with body:
            english = item.get("title_english")
            subtitle = f" · *{english}*" if english and english != item["title"] else ""
            st.markdown(f"#### [{item['title']}]({item['mal_url']}){subtitle}")
            episodes = f"{item['episodes']} ep" if item.get("episodes") else "ongoing"
            aired = item.get("season") or item.get("start_year") or "?"
            facts = [f"⭐ **{item['score']:.2f}**", f"#{item['rank']}", item["type"], episodes,
                     str(aired), f"{item['members']:,} members"]  # fmt: skip
            if item.get("studios"):
                facts.append(", ".join(item["studios"][:2]))
            if item.get("source"):
                facts.append(f"from {item['source']}")
            st.markdown(" · ".join(facts))
            tags = item["genres"] + item["themes"] + item["demographics"]
            if tags:
                st.markdown(" ".join(f":gray-background[{t}]" for t in tags))
            if item.get("reason"):
                st.markdown(f"💡 {item['reason']}")
            elif item.get("synopsis"):
                st.markdown(
                    (item["synopsis"][:260] + "…")
                    if len(item["synopsis"]) > 260
                    else item["synopsis"]
                )
            with st.expander("Synopsis & characters"):
                st.write(item.get("synopsis") or "No synopsis available.")
                if item.get("directors"):
                    st.caption("Directed by " + ", ".join(item["directors"]))
                chars = item.get("main_characters", [])[:CHARACTERS_SHOWN]
                if chars:
                    for col, ch in zip(st.columns(len(chars)), chars, strict=True):
                        if ch.get("image_url"):
                            col.image(ch["image_url"], width="stretch")
                        col.caption(
                            f"**{ch['name']}**"
                            + (f"  \nCV: {ch['voice_actor']}" if ch.get("voice_actor") else "")
                        )
            b1, b2, _ = st.columns([1, 1, 3])
            b1.button("More like this", key=f"more-{key}", on_click=more_like,
                      args=(item["anime_id"], item["title"]))  # fmt: skip
            b2.button("👍 I liked it", key=f"like-{key}", on_click=like, args=(item["anime_id"],))


def render_answer(response: dict[str, Any], turn: int) -> None:
    render_understood(response.get("understood"), response.get("seeds"))
    if response.get("summary"):
        st.markdown(response["summary"])
    if not response["items"]:
        st.info("Nothing matched. Try loosening the filters.")
    for i, item in enumerate(response["items"]):
        render_card(item, key=f"{turn}-{i}")
    note = "re-ranked and explained by Gemini" if response.get("explained") else "vector search"
    st.caption(f"{response['took_ms']} ms · {note}")


# --- conversation ---------------------------------------------------------------------

st.title("🎌 Anime Recommender")
st.caption("Describe what you feel like watching. Titles, moods, lengths and eras all work.")

prompt = st.chat_input("e.g. a dark psychological thriller with a smart protagonist")
if not state.messages and not prompt and not state.pending:
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES, strict=True):
        if col.button(example, width="stretch"):
            prompt = example

for turn, msg in enumerate(state.messages):
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎌"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif "error" in msg:
            st.error(msg["error"])
        else:
            render_answer(msg["response"], turn)

request: dict[str, Any] | None = None
if prompt:
    request = {"kind": "chat", "text": prompt}
elif state.pending:
    request, state.pending = state.pending, None
    request["text"] = f"More like {request['title']}"

if request:
    turn = len(state.messages)
    state.messages.append({"role": "user", "content": request["text"]})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(request["text"])
    with st.chat_message("assistant", avatar="🎌"), st.spinner("Finding anime…"):
        try:
            common = {"filters": filters, "limit": limit, "explain": explain}
            if request["kind"] == "similar":
                response = api.similar(request["anime_id"], **common)
            else:
                response = api.chat(
                    request["text"],
                    liked=[title_ids[x] for x in state.liked],
                    disliked=[title_ids[x] for x in state.disliked],
                    **common,
                )
        except api_client.ApiError as exc:
            state.messages.append({"role": "assistant", "error": str(exc)})
            st.error(str(exc))
        else:
            state.messages.append({"role": "assistant", "response": response})
            render_answer(response, turn + 1)
