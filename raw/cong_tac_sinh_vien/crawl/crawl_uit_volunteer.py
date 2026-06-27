from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

PARENT_DOCUMENT = "Tình nguyện tại UIT"
OUTPUT_PATH = "uit_volunteer_2025_2026.json"

YEARS = {2025, 2026}

LIST_PAGE_URLS = [
    "https://tuoitre.uit.edu.vn/category/phong-trao-tinh-nguyen",
]

# Seed thêm các bài chính thức của UIT mà category Tuổi trẻ UIT có thể không liệt kê đủ.
SEED_ARTICLE_URLS = [
    "https://www.uit.edu.vn/tong-quan-chien-dich-xuan-tinh-nguyen-2026-lan-thu-18",
    "https://www.uit.edu.vn/bai-viet/dang-ky-tro-thanh-chien-si-xuan-tinh-nguyen-lan-thu-18-nam-2026-cua-hoi-sinh-vien-viet-nam-tp-ho-chi-minh",
    "https://www.uit.edu.vn/bai-viet/dang-ky-tro-thanh-chien-si-mua-he-xanh-uit-2026",
    "https://www.uit.edu.vn/bai-viet/ra-mat-ban-chi-huy-cac-doi-hinh-mua-he-xanh-uit-2026",
    "https://www.uit.edu.vn/bai-viet/giai-ma-doi-hinh-thuong-truc-mua-he-xanh-uit-2026",
    "https://www.uit.edu.vn/bai-viet/giai-ma-doi-hinh-chuyen-mua-he-xanh-uit-2026",
    "https://ctsv.uit.edu.vn/node/456",
]

VOLUNTEER_KEYWORDS = [
    "tình nguyện",
    "xuan tinh nguyen",
    "xuân tình nguyện",
    "mùa hè xanh",
    "mua he xanh",
    "chủ nhật xanh",
    "chu nhat xanh",
    "uit green",
    "hiến máu",
    "hien mau",
    "giọt hồng",
    "giot hong",
    "trung thu yêu thương",
    "trung thu yeu thuong",
    "hướng về miền trung",
    "huong ve mien trung",
    "tiếp sức",
    "tiep suc",
    "chiến sĩ",
    "chien si",
    "cộng đồng",
    "cong dong",
]

ALLOWED_DOMAINS = {
    "tuoitre.uit.edu.vn",
    "www.uit.edu.vn",
    "uit.edu.vn",
    "ctsv.uit.edu.vn",
}


def normalize_space(text: str | None) -> str:
    text = text or ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def normalize_text(text: str | None) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")

    lines = []
    prev_blank = False
    for line in text.splitlines():
        line = normalize_space(line)
        if not line:
            if not prev_blank:
                lines.append("")
            prev_blank = True
            continue

        lines.append(line)
        prev_blank = False

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def strip_accents_for_match(text: str) -> str:
    # Không dùng thư viện ngoài; đủ tốt cho matching keyword tiếng Việt cơ bản.
    replacements = {
        "àáạảãâầấậẩẫăằắặẳẵ": "a",
        "èéẹẻẽêềếệểễ": "e",
        "ìíịỉĩ": "i",
        "òóọỏõôồốộổỗơờớợởỡ": "o",
        "ùúụủũưừứựửữ": "u",
        "ỳýỵỷỹ": "y",
        "đ": "d",
    }
    output = text.lower()
    for chars, replacement in replacements.items():
        for ch in chars:
            output = output.replace(ch, replacement)
    return output


def contains_volunteer_keyword(text: str) -> bool:
    raw = text.lower()
    folded = strip_accents_for_match(text)

    for kw in VOLUNTEER_KEYWORDS:
        kw_raw = kw.lower()
        kw_folded = strip_accents_for_match(kw)
        if kw_raw in raw or kw_folded in folded:
            return True

    return False


def make_absolute_url(href: str, base_url: str) -> str:
    href = normalize_space(href)
    if not href:
        return ""

    if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
        return ""

    return urljoin(base_url, href)


def is_allowed_url(url: str) -> bool:
    if not url:
        return False

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False

    return parsed.netloc.lower() in ALLOWED_DOMAINS


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
    }

    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    if response.encoding is None:
        response.encoding = response.apparent_encoding

    return response.text


def get_meta_content(soup: BeautifulSoup, *names: str) -> str:
    for name in names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return normalize_space(tag["content"])
    return ""


def extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        title = normalize_space(h1.get_text(" ", strip=True))
        if title:
            return title

    meta_title = get_meta_content(soup, "og:title", "twitter:title")
    if meta_title:
        return meta_title

    if soup.title:
        return normalize_space(soup.title.get_text(" ", strip=True))

    return ""


def extract_date(text: str, soup: BeautifulSoup) -> tuple[str, int | None]:
    # Ưu tiên meta date nếu có.
    meta_date = get_meta_content(
        soup,
        "article:published_time",
        "article:modified_time",
        "date",
        "pubdate",
        "publishdate",
    )

    candidates = []
    if meta_date:
        candidates.append(meta_date)

    candidates.append(text)

    for candidate in candidates:
        # 02/01/2026, 02-01-2026
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", candidate)
        if m:
            date_text = f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"
            return date_text, int(m.group(3))

        # 2026-01-02 hoặc ISO datetime
        m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", candidate)
        if m:
            date_text = f"{int(m.group(3)):02d}/{int(m.group(2)):02d}/{m.group(1)}"
            return date_text, int(m.group(1))

        # fallback: lấy năm trong title/content
        m = re.search(r"\b(2025|2026)\b", candidate)
        if m:
            return "", int(m.group(1))

    return "", None


def clean_html_node(node, base_url: str) -> str:
    node_copy = BeautifulSoup(str(node), "html.parser")

    for tag in node_copy.find_all([
        "script",
        "style",
        "noscript",
        "svg",
        "form",
        "iframe",
        "button",
        "nav",
        "header",
        "footer",
        "aside",
    ]):
        tag.decompose()

    for img in node_copy.find_all("img"):
        alt = normalize_space(img.get("alt", ""))
        if alt and not alt.lower().startswith("image"):
            img.replace_with(alt)
        else:
            img.decompose()

    for a in node_copy.find_all("a"):
        href = make_absolute_url(a.get("href", ""), base_url)
        label = normalize_space(a.get_text(" ", strip=True))

        if href and href.startswith(("http://", "https://")):
            replacement = f"{label} ({href})" if label and label != href else href
            a.replace_with(replacement)
        else:
            a.replace_with(label)

    return normalize_text(node_copy.get_text("\n", strip=True))


def select_main_content(soup: BeautifulSoup):
    selectors = [
        "article .entry-content",
        "article .post-content",
        "article .field--name-body",
        "article .node__content",
        "article",
        ".entry-content",
        ".post-content",
        ".field--name-body",
        ".node__content",
        ".region-content article",
        ".region-content",
        "main article",
        "main",
    ]

    for selector in selectors:
        node = soup.select_one(selector)
        if node and normalize_space(node.get_text(" ", strip=True)):
            return node

    return soup.body or soup


def discover_article_links_from_list_page(url: str) -> set[str]:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    links: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = make_absolute_url(a.get("href", ""), url)
        label = normalize_space(a.get_text(" ", strip=True))

        if not is_allowed_url(href):
            continue

        # Bỏ chính trang category, feed, tag, page archive.
        if any(x in href for x in ["/category/", "/tag/", "/feed", "#"]):
            continue

        # Lấy rộng các link trong category; bài detail sẽ filter keyword/year sau.
        if len(label) >= 10 or contains_volunteer_keyword(href):
            links.add(href.split("#")[0])

    return links


def build_content(title: str, date_text: str, body: str) -> str:
    body = normalize_text(body)

    # Nếu body đã bắt đầu bằng title thì không lặp title.
    if title and not body.lower().startswith(title.lower()):
        prefix = title
        if date_text:
            prefix += f". Bài viết được đăng ngày {date_text}"
        prefix += "."
        return normalize_text(prefix + "\n\n" + body)

    if date_text and "Bài viết được đăng ngày" not in body:
        return normalize_text(f"Bài viết được đăng ngày {date_text}.\n\n{body}")

    return body


def parse_article(url: str) -> dict | None:
    try:
        html = fetch_html(url)
    except Exception as exc:
        print(f"Skip fetch failed: {url} | {exc}")
        return None

    soup = BeautifulSoup(html, "html.parser")

    title = extract_title(soup)
    main_node = select_main_content(soup)
    body = clean_html_node(main_node, url)
    full_text_for_match = f"{title}\n{body}"

    date_text, year = extract_date(full_text_for_match, soup)

    if year not in YEARS:
        return None

    if not contains_volunteer_keyword(full_text_for_match):
        return None

    content = build_content(title, date_text, body)

    if len(content) < 80:
        return None

    return {
        "link": url,
        "title": title,
        "date": date_text,
        "year": year,
        "content": content,
    }


def crawl(max_list_pages: int) -> list[dict]:
    candidate_urls: set[str] = set(SEED_ARTICLE_URLS)

    # Crawl list page + một số page pagination phổ biến.
    for base_url in LIST_PAGE_URLS:
        candidate_urls.update(discover_article_links_from_list_page(base_url))

        for page in range(2, max_list_pages + 1):
            for paged_url in [
                f"{base_url}/page/{page}",
                f"{base_url}?page={page}",
                f"{base_url}?paged={page}",
            ]:
                try:
                    links = discover_article_links_from_list_page(paged_url)
                except Exception:
                    continue
                candidate_urls.update(links)

    parsed_items = []
    seen_urls = set()

    for url in sorted(candidate_urls):
        if url in seen_urls:
            continue

        seen_urls.add(url)

        item = parse_article(url)
        if item:
            parsed_items.append(item)

    # Sort theo year/date/title cho ổn định.
    parsed_items.sort(key=lambda x: (x["year"], x["date"], x["title"], x["link"]))

    records = []
    for idx, item in enumerate(parsed_items, start=1):
        records.append({
            "id": idx,
            "link": item["link"],
            "parent_document": PARENT_DOCUMENT,
            "content": item["content"],
            "metadata": {
                "category": "tinh_nguyen",
                "major": None,
                "program": None,
                "year": item["year"],
                "status": None,
            },
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="File JSON output.",
    )
    parser.add_argument(
        "--max-list-pages",
        type=int,
        default=3,
        help="Số trang category tối đa cần thử crawl.",
    )
    args = parser.parse_args()

    records = crawl(max_list_pages=args.max_list_pages)

    Path(args.output).write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(records)} records to {args.output}")

    by_year: dict[int, int] = {}
    for record in records:
        year = record["metadata"]["year"]
        by_year[year] = by_year.get(year, 0) + 1

    print("Records by year:", by_year)


if __name__ == "__main__":
    main()
