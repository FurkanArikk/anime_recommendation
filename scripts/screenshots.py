"""Drive the running web UI in headless Chromium and save screenshots (for docs + a visual
smoke test). Any console error or uncaught page error fails the run.

    make screenshots        # stack must be up (make up)
"""

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://web"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/out")
PROMPT = "Something like Attack on Titan but funnier, under 25 episodes"


def wait_for_images(page: Page) -> None:
    page.wait_for_function(
        "() => [...document.images].filter(i => i.getBoundingClientRect().top < innerHeight)"
        ".every(i => i.complete)",
        timeout=20_000,
    )


def main() -> None:
    errors: list[str] = []
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.on(
            "console",
            lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else None,
        )
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        page.goto(BASE, wait_until="networkidle")
        page.wait_for_selector("#status.ok", timeout=20_000)
        wait_for_images(page)
        page.screenshot(path=OUT / "1-home.png")

        page.fill("#message", PROMPT)
        page.keyboard.press("Enter")
        page.wait_for_selector(".turn-bot .card:not(.skeleton)", timeout=60_000)
        page.wait_for_timeout(1200)  # smooth scroll to the answer
        wait_for_images(page)
        page.screenshot(path=OUT / "2-results.png")

        page.click(".turn-bot .card:not(.skeleton) .poster")
        page.wait_for_selector("dialog.detail[open]")
        page.wait_for_timeout(500)
        page.screenshot(path=OUT / "3-detail.png")
        page.keyboard.press("Escape")

        page.click("#filters-btn")
        page.click("#tag-chips .toggle >> nth=1")  # include
        page.click("#tag-chips .toggle >> nth=4")
        page.click("#tag-chips .toggle >> nth=4")  # exclude
        page.wait_for_timeout(300)
        page.screenshot(path=OUT / "4-filters.png")
        page.click("#reset-filters")
        page.keyboard.press("Escape")

        page.click("#taste-btn")
        page.fill("#title-search", "frieren")
        page.wait_for_selector(".suggestion")
        wait_for_images(page)
        page.screenshot(path=OUT / "5-taste.png")
        page.keyboard.press("Escape")

        page.click("#new-chat")
        page.fill("#message", "I watched Haikyuu and loved it, what else should I watch?")
        page.keyboard.press("Enter")
        page.wait_for_selector(".turn-bot .franchise", timeout=60_000)
        page.wait_for_selector(".turn-bot .card:not(.skeleton)", timeout=60_000)
        page.wait_for_timeout(1200)
        wait_for_images(page)
        page.screenshot(path=OUT / "7-franchise.png")

        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        mobile.on("pageerror", lambda e: errors.append(f"mobile pageerror: {e}"))
        mobile.goto(BASE, wait_until="networkidle")
        mobile.fill("#message", "cozy slice of life")
        mobile.keyboard.press("Enter")
        mobile.wait_for_selector(".turn-bot .card:not(.skeleton)", timeout=60_000)
        mobile.wait_for_timeout(1200)
        mobile.screenshot(path=OUT / "6-mobile.png")
        browser.close()

    print("\n".join(errors) or "no browser errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
