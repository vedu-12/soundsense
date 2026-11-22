"""Launch the app headless and screenshot every page with Playwright (desktop width).

Usage: python tests/take_screenshots.py [out_dir]   (default docs/screenshots)
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "screenshots"
PORT = 8599
PAGES = {"0_home": "", "1_song_map": "Song_Map", "2_happiness_meter": "Happiness_Meter",
         "3_genre_detector": "Genre_Detector", "4_similar_songs": "Similar_Songs", "5_hit_predictor": "Hit_Predictor",
         "6_song_lab": "Song_Lab_and_Model_Arena", "7_mood_playlists": "Mood_Playlists"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/Home.py", "--server.port", str(PORT),
                             "--server.headless", "true"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            for name, path in PAGES.items():
                page.goto(f"http://localhost:{PORT}/{path}", wait_until="networkidle")
                page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=30000)
                time.sleep(6)                                   # let charts render
                errs = page.locator("[data-testid='stException']").count()
                # Streamlit scrolls inside its main container: grow the viewport to the content height first
                h = page.evaluate("() => { const m = document.querySelector('[data-testid=\"stMain\"]'); return m ? m.scrollHeight : 900 }")
                page.set_viewport_size({"width": 1440, "height": max(900, min(int(h) + 40, 6000))})
                time.sleep(2)
                page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
                page.set_viewport_size({"width": 1440, "height": 900})
                print(f"{name}: saved{'  (EXCEPTION ON PAGE)' if errs else ''}")
                if name == "6_song_lab":                       # second tab
                    page.get_by_role("tab", name="Model Arena").click()
                    time.sleep(4)
                    h = page.evaluate("() => { const m = document.querySelector('[data-testid=\"stMain\"]'); return m ? m.scrollHeight : 900 }")
                    page.set_viewport_size({"width": 1440, "height": max(900, min(int(h) + 40, 6000))})
                    time.sleep(2)
                    page.screenshot(path=str(OUT / "6b_model_arena.png"), full_page=True)
                    page.set_viewport_size({"width": 1440, "height": 900})
                    print("6b_model_arena: saved")
            browser.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
