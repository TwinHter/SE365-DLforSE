import os
import re
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUTPUT_DIR = "html_pages"

URLS = {
    "nlp_research": {
        "url": "https://nlp.uit.edu.vn/research",
        "verify": True,
    },
    "nlp_members": {
        "url": "https://nlp.uit.edu.vn/members",
        "verify": True,
    },
    "datachain_research_group": {
        "url": "https://datachain.uit.edu.vn/nhom-nghien-cuu/",
        "verify": True,
    },
    "inseclab_about": {
        "url": "https://inseclab.uit.edu.vn/gioi-thieu/",
        "verify": False,
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
}


def safe_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    return name.strip("_")


def download_html(name: str, config: dict) -> None:
    url = config["url"]
    verify = config.get("verify", True)

    filename = safe_filename(name) + ".html"
    output_path = os.path.join(OUTPUT_DIR, filename)

    print(f"Downloading: {url}")

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
        allow_redirects=True,
        verify=verify,
    )

    response.raise_for_status()

    if response.encoding is None:
        response.encoding = response.apparent_encoding

    html = response.text

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved: {output_path}")
    print(f"Final URL: {response.url}")
    print(f"Size: {len(html):,} chars")
    print("-" * 60)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, config in URLS.items():
        try:
            download_html(name, config)
        except Exception as e:
            print(f"FAILED: {name} - {config['url']}")
            print(f"Error: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()