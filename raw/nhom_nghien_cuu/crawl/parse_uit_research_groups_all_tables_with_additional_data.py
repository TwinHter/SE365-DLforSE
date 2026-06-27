from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pathlib import Path
import re
import html
import json

INPUT_PATH = Path("html_pages") / "fit_research_uit.html"
OUTPUT_PATH = "uit_research_groups_2026.json"
ADDITIONAL_DATA_PATH = "research_website_additional_content.json"

PARENT_DOCUMENT = "Các nhóm nghiên cứu UIT 2026"
INCLUDE_SOURCE_FIELDS = True
ENRICH_ADDITIONAL_CONTENT = True
# Chỉ gắn additional data cho các record có URL/link khớp trong nội dung.
# Không gắn theo tên nhóm nghiên cứu để tránh enrich nhầm các dòng không có link.
ENRICH_ONLY_WHEN_RECORD_HAS_LINK = True

PROTECTED_TEXT = "This email address is being protected from spambots. You need JavaScript enabled to view it."


def decode_joomla_emails(tag):
    """
    Decode Joomla anti-spam email scripts like:
      var addy123 = 'abc' + '&#64;';
      addy123 = addy123 + 'uit' + '&#46;' + 'edu' + '&#46;' + 'vn';
    """
    emails = []

    for script in tag.find_all("script"):
        txt = script.string or script.get_text() or ""
        lines = [line.strip() for line in txt.splitlines() if line.strip()]

        for idx, line in enumerate(lines):
            m = re.match(r"var\s+(addy\d+)\s*=\s*(.+);", line)
            if not m:
                continue

            var_name, expr1 = m.group(1), m.group(2)
            expr2 = ""

            for next_line in lines[idx + 1 : idx + 8]:
                m2 = re.match(
                    rf"{re.escape(var_name)}\s*=\s*{re.escape(var_name)}\s*\+\s*(.+);",
                    next_line,
                )
                if m2:
                    expr2 = m2.group(1)
                    break

            def eval_concat(expr):
                parts = re.findall(r"'([^']*)'", expr)
                return html.unescape("".join(parts))

            email = (eval_concat(expr1) + eval_concat(expr2)).strip()

            if "@" in email and "." in email and email not in emails:
                emails.append(email)

    return emails


def normalize_space(text):
    text = html.unescape(text or "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def post_process_lines(lines):
    """
    Merge artifacts caused by inline spans or punctuation-only nodes.
    Example: ["2. ... xã hội,", "giáo dục, ..."] -> one line.
    """
    output = []

    for line in lines:
        line = normalize_space(line)
        if not line:
            continue

        if line == "." and output:
            output[-1] = output[-1].rstrip() + "."
            continue

        should_merge = False
        if output:
            prev = output[-1]
            starts_new_item = re.match(
                r"^(Email|Website|Wedsite|GG\s*Scholar|GG\s*scholar|Yêu cầu|Lưu ý|\d+\.|[ivx]+\)|[-+•])\b",
                line,
                flags=re.I,
            )

            if not starts_new_item:
                if prev.endswith((",", ":", "/", ";")):
                    should_merge = True
                elif re.match(r"^[a-zà-ỹ]", line):
                    should_merge = True

        if should_merge:
            output[-1] = normalize_space(output[-1] + " " + line)
        else:
            output.append(line)

    return output


def clean_text_from_tag(tag, keep_links=True):
    """
    Return cleaned text lines from an HTML tag.
    - Removes scripts and protected email placeholder text.
    - Preserves HTTP links as "anchor text (url)".
    - Does not decode Joomla emails here; use decode_joomla_emails separately.
    """
    tag_copy = BeautifulSoup(str(tag), "html.parser")

    for script in tag_copy.find_all("script"):
        script.decompose()

    if keep_links:
        for a in tag_copy.find_all("a"):
            href = normalize_space(a.get("href", ""))
            label = normalize_space(a.get_text(" ", strip=True))

            if href and href.lower().startswith(("http://", "https://")):
                replacement = f"{label} ({href})" if label and label != href else href
                a.replace_with(replacement)
            elif href and href.lower().startswith("mailto:"):
                a.replace_with(label)
            else:
                a.replace_with(label)

    text = tag_copy.get_text("\n", strip=True)
    text = text.replace(PROTECTED_TEXT, "")
    text = text.replace("\xa0", " ")

    lines = []
    for line in text.splitlines():
        line = normalize_space(line)
        line = line.replace("---------------------------------------", "").strip()
        if line:
            lines.append(line)

    return post_process_lines(lines)


def clean_scalar_text(tag):
    return " ".join(clean_text_from_tag(tag)).strip() or None


def strip_list_marker(text):
    text = normalize_space(text)
    text = re.sub(r"^[•\-+]\s*", "", text)
    return text.strip()


def clean_list_text(tag, append_email=False):
    """
    Prefer <li> items when available. Otherwise return cleaned text lines.
    """
    items = []

    for li in tag.find_all("li"):
        lines = clean_text_from_tag(li)
        item = " ".join(lines).strip()
        item = strip_list_marker(item)
        if item:
            items.append(item)

    if not items:
        for line in clean_text_from_tag(tag):
            line = strip_list_marker(line)
            if line:
                items.append(line)

    if append_email:
        emails = decode_joomla_emails(tag)
        if emails:
            new_items = []
            inserted = False

            for line in items:
                if re.fullmatch(r"-?\s*Email\s*:?\s*", line, flags=re.I):
                    if not inserted:
                        new_items.append(f"Email: {', '.join(emails)}")
                        inserted = True
                    continue

                if re.match(r"^-?\s*Email\s*:", line, flags=re.I) and "@" not in line:
                    if not inserted:
                        new_items.append(f"Email: {', '.join(emails)}")
                        inserted = True
                    continue

                new_items.append(line)

            if not inserted:
                new_items.append(f"Email: {', '.join(emails)}")

            items = new_items

    return items


def split_major_and_group(text):
    text = text.replace("\xa0", " ")
    major = None
    group = None

    m_major = re.search(r"Bộ môn\s*:\s*(.+?)(?:\n|$)", text, flags=re.I)
    if m_major:
        major = normalize_space(m_major.group(1))

    m_group = re.search(r"Nhóm nghiên cứu\s*:\s*(.+?)(?:\n|$)", text, flags=re.I)
    if m_group:
        group = normalize_space(m_group.group(1))

    return major or None, group or None


def split_note_and_requirement(note_lines):
    """
    Some old tables put "Yêu cầu: ..." inside the note column.
    Move those lines into content["Yêu cầu"].
    """
    notes = []
    requirements = []
    in_requirement_block = False

    for line in note_lines:
        raw = normalize_space(line)
        if not raw:
            continue

        m = re.match(r"^-?\s*Yêu cầu(?:\s+về\s+phía\s+sinh\s+viên)?\s*:?\s*(.*)$", raw, flags=re.I)
        if m:
            in_requirement_block = True
            rest = normalize_space(m.group(1))
            if rest:
                requirements.append(rest)
            continue

        if re.match(r"^Lưu ý\s*:", raw, flags=re.I):
            in_requirement_block = False
            notes.append(raw)
            continue

        if in_requirement_block:
            requirements.append(raw)
        else:
            notes.append(raw)

    return notes, requirements


def make_additional_content(parts):
    """
    additional_content là list.
    Các thông tin ngoài schema chính từ bảng HTML được giữ ở 1 phần tử text riêng.
    Các trang website crawl thêm sẽ được append thành các phần tử tiếp theo.
    """
    lines = []
    for title, values in parts:
        if values is None:
            continue

        if isinstance(values, str):
            values = [values] if values.strip() else []

        values = [normalize_space(v) for v in values if normalize_space(v)]
        if not values:
            continue

        lines.append(f"{title}:")
        for value in values:
            lines.append(f"- {value}")

    return ["\n".join(lines)] if lines else []


def is_data_row(cells):
    if not cells:
        return False
    first = clean_scalar_text(cells[0]) or ""
    return bool(re.search(r"\d+", first))


def find_header_row(rows):
    """
    Find the header row because some tables have a section-title row before headers.
    """
    for idx, tr in enumerate(rows):
        cells = tr.find_all(["td", "th"])
        row_text = " | ".join(clean_scalar_text(cell) or "" for cell in cells).lower()

        has_stt = "stt" in row_text
        has_teacher = "giảng viên" in row_text or "người có thể hướng dẫn" in row_text
        has_research = "hướng nghiên cứu" in row_text or "lĩnh vực nc" in row_text

        if has_stt and has_teacher and has_research:
            return idx

    return None


def get_table_section_title(table, table_index):
    """
    Best-effort section label for traceability.
    Not required by the main schema, so it is stored only in metadata.source_section.
    """
    first_row = table.find("tr")
    if first_row:
        first_cells = first_row.find_all(["td", "th"])
        if len(first_cells) == 1:
            title = clean_scalar_text(first_cells[0])
            if title and not re.fullmatch(r"STT", title, flags=re.I):
                return title

    prev = table.find_previous(["p", "h1", "h2", "h3"])
    if prev:
        title = clean_scalar_text(prev)
        if title:
            return title

    return f"Bảng {table_index + 1}"


def detect_table_type(headers):
    header_text = " | ".join(headers).lower()

    if "bộ môn và nhóm nghiên cứu" in header_text:
        return "faculty_research_groups_2026"

    if "chủ đề nghiên cứu" in header_text:
        return "research_topics_kltn"

    if "lĩnh vực nc" in header_text or "số lượng ncs" in header_text or "người có thể hướng dẫn" in header_text:
        return "postgraduate_research_directions"

    return "unknown"


def build_record(record_id, lecturer, research_group, research_directions, requirements, notes, major, additional_content, source_section, table_type):
    record = {
        "id": record_id,
        "parent_document": PARENT_DOCUMENT,
        "content": {
            "Giảng viên hướng dẫn": lecturer or None,
            "nhóm nghiên cứu": research_group or None,
            "Hướng nghiên cứu/ứng dụng": research_directions or [],
            "Yêu cầu": requirements or [],
            "Ghi chú": notes or [],
            "additional_content": additional_content or [],
        },
        "metadata": {
            "category": "nghien_cuu",
            "major": major or None,
            "program": None,
            "year": 2026,
            "status": "Active",
        },
    }

    if INCLUDE_SOURCE_FIELDS:
        record["metadata"]["source_section"] = source_section
        record["metadata"]["source_table_type"] = table_type

    return record


def parse_table_2026(cells, record_id, source_section, table_type):
    lecturer = clean_scalar_text(cells[1])

    major_group_text = "\n".join(clean_text_from_tag(cells[2]))
    major, research_group = split_major_and_group(major_group_text)

    research_directions = clean_list_text(cells[3])
    requirements = clean_list_text(cells[4])
    notes = clean_list_text(cells[5], append_email=True)

    return build_record(
        record_id=record_id,
        lecturer=lecturer,
        research_group=research_group,
        research_directions=research_directions,
        requirements=requirements,
        notes=notes,
        major=major,
        additional_content=[],
        source_section=source_section,
        table_type=table_type,
    )


def parse_kltn_table(cells, record_id, source_section, table_type):
    lecturer = clean_scalar_text(cells[1])
    research_directions = clean_list_text(cells[2])
    research_topics = clean_list_text(cells[3])
    note_lines = clean_list_text(cells[4], append_email=True)

    notes, extracted_requirements = split_note_and_requirement(note_lines)

    additional_content = make_additional_content([
        ("Chủ đề nghiên cứu", research_topics),
    ])

    return build_record(
        record_id=record_id,
        lecturer=lecturer,
        research_group=None,
        research_directions=research_directions,
        requirements=extracted_requirements,
        notes=notes,
        major=None,
        additional_content=additional_content,
        source_section=source_section,
        table_type=table_type,
    )


def parse_postgraduate_table(cells, record_id, source_section, table_type):
    research_directions = clean_list_text(cells[1])
    lecturer = clean_scalar_text(cells[2])
    requirements = clean_list_text(cells[3])

    support = clean_list_text(cells[4])
    ncs_count = clean_scalar_text(cells[5])
    other_requirements = clean_list_text(cells[6])

    additional_content = make_additional_content([
        ("Các chế độ hỗ trợ NCS", support),
        ("Số lượng NCS còn có thể nhận", [ncs_count] if ncs_count else []),
        ("Các yêu cầu khác", other_requirements),
    ])

    return build_record(
        record_id=record_id,
        lecturer=lecturer,
        research_group=None,
        research_directions=research_directions,
        requirements=requirements,
        notes=[],
        major=None,
        additional_content=additional_content,
        source_section=source_section,
        table_type=table_type,
    )


def parse_all_tables(soup):
    records = []

    main_content = soup.select_one("#system") or soup
    tables = main_content.find_all("table")

    for table_index, table in enumerate(tables):
        rows = table.find_all("tr")
        header_idx = find_header_row(rows)

        if header_idx is None:
            continue

        header_cells = rows[header_idx].find_all(["td", "th"])
        headers = [clean_scalar_text(cell) or "" for cell in header_cells]
        table_type = detect_table_type(headers)

        if table_type == "unknown":
            continue

        source_section = get_table_section_title(table, table_index)

        for tr in rows[header_idx + 1:]:
            cells = tr.find_all(["td", "th"])

            if not is_data_row(cells):
                continue

            try:
                record_id = len(records) + 1

                if table_type == "faculty_research_groups_2026":
                    if len(cells) < 6:
                        continue
                    record = parse_table_2026(cells, record_id, source_section, table_type)

                elif table_type == "research_topics_kltn":
                    if len(cells) < 5:
                        continue
                    record = parse_kltn_table(cells, record_id, source_section, table_type)

                elif table_type == "postgraduate_research_directions":
                    if len(cells) < 7:
                        continue
                    record = parse_postgraduate_table(cells, record_id, source_section, table_type)

                else:
                    continue

                if record["content"]["Giảng viên hướng dẫn"]:
                    records.append(record)

            except Exception as exc:
                print(f"Skip row in table {table_index + 1} because of parse error: {exc}")

    return records


def normalize_url(url):
    url = normalize_space(url)
    url = url.strip("'\"<>[]{}.,;。")
    url = re.sub(r"[)]+$", "", url)
    return url


def walk_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_values(child)
    elif isinstance(value, str):
        yield value


def extract_urls_from_record(record):
    text = "\n".join(walk_values(record.get("content", {})))
    candidates = re.findall(r"https?://[^\s\)\]\}\"'<>]+", text)
    urls = []

    for url in candidates:
        url = normalize_url(url)
        if not url:
            continue
        if url not in urls:
            urls.append(url)

    return urls


def canonical_url_key(url):
    url = normalize_url(url)
    return url[:-1] if url.endswith("/") else url



# Static additional-content enrichment.
# This version DOES NOT crawl external websites while parsing.
# It only attaches curated text from ADDITIONAL_DATA_PATH.

def load_static_additional_data():
    path = Path(ADDITIONAL_DATA_PATH)
    if not path.exists():
        print(f"Static additional data file not found: {ADDITIONAL_DATA_PATH}. Skip enrichment.")
        return []

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        return payload.get("entries", [])

    if isinstance(payload, list):
        return payload

    raise ValueError("Invalid additional data JSON format. Expected object with key 'entries' or a list.")


def lower_text(value):
    return normalize_space(value).lower()


def canonical_match_url(url):
    key = canonical_url_key(url).lower()
    parsed = urlparse(key)
    if parsed.scheme and parsed.netloc:
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{parsed.netloc}{path}"
    return key.rstrip("/")


def urls_match(record_url, match_url):
    """
    Match exact URL or same host root.
    Example:
      record_url: http://nlp.uit.edu.vn
      match_url:  https://nlp.uit.edu.vn/
    Both should match by host.
    """
    r = canonical_match_url(record_url)
    m = canonical_match_url(match_url)

    if r == m:
        return True

    parsed_r = urlparse(r)
    parsed_m = urlparse(m)

    if parsed_r.netloc and parsed_m.netloc and parsed_r.netloc == parsed_m.netloc:
        # If the matching URL is the domain root, match any URL from the same host.
        return parsed_m.path in ("", "/")

    return False


def get_matched_urls_for_entry(record, entry):
    """
    Return URLs in this record that match an entry in the static additional-data file.
    Matching is URL-based only, because we only want to enrich records that actually contain a link.
    """
    record_urls = extract_urls_from_record(record)
    match_urls = entry.get("match_urls", []) or []
    matched = []

    for record_url in record_urls:
        for match_url in match_urls:
            if urls_match(record_url, match_url):
                if record_url not in matched:
                    matched.append(record_url)
                break

    return matched


def record_matches_static_entry(record, entry):
    return bool(get_matched_urls_for_entry(record, entry))


def format_static_content_item(item):
    if isinstance(item, str):
        return normalize_space(item)

    if not isinstance(item, dict):
        return ""

    lines = ["Thông tin bổ sung từ file static:"]

    source_page = normalize_space(item.get("source_page", ""))
    title = normalize_space(item.get("title", ""))
    text = normalize_space(item.get("text", ""))

    if source_page:
        lines.append(f"Trang nguồn: {source_page}")
    if title:
        lines.append(f"Tiêu đề: {title}")
    if text:
        lines.append(text)

    return "\n".join(lines).strip()


def enrich_records_with_static_additional_content(records):
    """
    Attach curated website information into content.additional_content.
    Each crawled/static page becomes one element in the additional_content array.
    Enrichment is URL-based only: a record must contain a matching URL in its parsed content.
    """
    report = []

    if not ENRICH_ADDITIONAL_CONTENT:
        return records, report

    entries = load_static_additional_data()
    if not entries:
        return records, report

    for record in records:
        additional = record["content"].get("additional_content") or []
        if isinstance(additional, str):
            additional = [additional] if additional.strip() else []

        seen = set(additional)
        record_report_items = []

        for entry in entries:
            matched_urls = get_matched_urls_for_entry(record, entry)
            if not matched_urls:
                continue

            added_sources = []
            for item in entry.get("additional_content", []) or []:
                text = format_static_content_item(item)
                if text and text not in seen:
                    additional.append(text)
                    seen.add(text)

                    if isinstance(item, dict):
                        added_sources.append(item.get("source_page") or item.get("title") or entry.get("name"))
                    else:
                        added_sources.append(entry.get("name"))

            if added_sources:
                record_report_items.append({
                    "entry_name": entry.get("name"),
                    "matched_urls": matched_urls,
                    "added_sources": added_sources,
                })

        record["content"]["additional_content"] = additional

        if record_report_items:
            report.append({
                "id": record.get("id"),
                "lecturer": record.get("content", {}).get("Giảng viên hướng dẫn"),
                "research_group": record.get("content", {}).get("nhóm nghiên cứu"),
                "source_section": record.get("metadata", {}).get("source_section"),
                "items": record_report_items,
            })

    return records, report


def resolve_input_path():
    candidates = [
        Path(INPUT_PATH),
        Path("fit_research_uit.html"),
        Path("html_pages") / "fit_research_uit.html",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Cannot find input HTML. Tried: " + ", ".join(str(path) for path in candidates)
    )


def print_enrichment_report(report):
    if not report:
        print("No records were enriched with static additional data.")
        return

    print("Enriched records:")
    for row in report:
        print(f"- id={row['id']} | GV={row['lecturer']} | nhóm={row['research_group']} | section={row['source_section']}")
        for item in row["items"]:
            print(f"  + matched entry: {item['entry_name']}")
            print(f"    matched urls: {', '.join(item['matched_urls'])}")
            print(f"    added sources: {', '.join(str(x) for x in item['added_sources'])}")


def main():
    input_path = resolve_input_path()

    with open(input_path, "r", encoding="utf-8") as f:
        raw_html = f.read()

    soup = BeautifulSoup(raw_html, "html.parser")
    records = parse_all_tables(soup)
    records, enrichment_report = enrich_records_with_static_additional_content(records)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Input: {input_path}")
    print(f"Saved {len(records)} records from all matched tables to {OUTPUT_PATH}")
    print_enrichment_report(enrichment_report)


if __name__ == "__main__":
    main()
