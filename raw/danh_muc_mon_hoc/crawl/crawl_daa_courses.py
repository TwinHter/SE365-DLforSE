from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

SOURCE_URL = "https://daa.uit.edu.vn/danh-muc-mon-hoc-dai-hoc"
PARENT_DOCUMENT = "Danh mục môn học đại học UIT"
DEFAULT_MAJOR_MAP_PATH = "major_code_name_map.json"


def normalize_space(text: str | None) -> str:
    text = text or ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def clean_cell_text(cell) -> str:
    """
    Lấy text trong một ô bảng.
    Với ô trạng thái, trang DAA dùng image alt/title như "Hiện đang mở" hoặc
    "Hiện không còn mở", nên cần thay <img> bằng alt/title trước khi get_text.
    """
    cell_copy = BeautifulSoup(str(cell), "html.parser")

    for tag in cell_copy.find_all(["script", "style"]):
        tag.decompose()

    for img in cell_copy.find_all("img"):
        status_text = (
            img.get("alt")
            or img.get("title")
            or img.get("data-original-title")
            or ""
        )
        img.replace_with(status_text)

    lines = []
    for line in cell_copy.get_text("\n", strip=True).splitlines():
        line = normalize_space(line)
        if line:
            lines.append(line)

    return "\n".join(lines)


def split_codes(text: str) -> list[str]:
    """
    Các cột mã tương đương / tiên quyết / môn học trước có thể chứa nhiều mã,
    mỗi mã nằm trên một dòng. Hàm này giữ lại danh sách mã sạch.
    """
    text = normalize_space(text)
    if not text:
        return []

    parts = re.split(r"[\n,;/]+", text)
    codes = []

    for part in parts:
        part = normalize_space(part)
        if not part:
            continue

        # Một số ô có thể bị dính text ngoài mã. Ưu tiên bắt token giống mã môn học.
        found = re.findall(r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b|\b[A-Z]+\d+\b", part)
        if found:
            codes.extend(found)
        else:
            codes.append(part)

    unique_codes = []
    seen = set()
    for code in codes:
        if code not in seen:
            unique_codes.append(code)
            seen.add(code)

    return unique_codes


def split_unit_codes(text: str) -> list[str]:
    """
    Tách mã ngành/đơn vị quản lý chuyên môn.
    Ví dụ: "KHMT", "MMT&TT", "HTTT / CNTT".
    """
    text = normalize_space(text)
    if not text:
        return []

    parts = re.split(r"[\n,;/]+", text)
    codes = []

    for part in parts:
        part = normalize_space(part)
        if part:
            codes.append(part)

    unique_codes = []
    seen = set()
    for code in codes:
        if code not in seen:
            unique_codes.append(code)
            seen.add(code)

    return unique_codes


def parse_int(value: str) -> int | None:
    value = normalize_space(value)
    if not value:
        return None

    m = re.search(r"-?\d+", value)
    return int(m.group(0)) if m else None


def is_open_status(text: str) -> bool | None:
    text = normalize_space(text).lower()
    if not text:
        return None

    if "không còn mở" in text or "khong con mo" in text:
        return False

    if "đang mở" in text or "dang mo" in text:
        return True

    return None


def load_json_map(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} phải là JSON object dạng mã: tên.")

    return {
        normalize_space(str(key)): normalize_space(str(value))
        for key, value in data.items()
        if normalize_space(str(key)) and normalize_space(str(value))
    }


def find_course_table(soup: BeautifulSoup):
    """
    Tìm bảng chứa danh mục môn học bằng header.
    """
    for table in soup.find_all("table"):
        header_text = normalize_space(table.get_text(" ", strip=True)).lower()
        if (
            "mã mh" in header_text
            and "tên mh" in header_text
            and "số tclt" in header_text
            and "số tcth" in header_text
        ):
            return table

    raise ValueError("Không tìm thấy bảng danh mục môn học trong HTML.")


def course_label(code: str, code_name_map: dict[str, str]) -> str:
    """
    Trả về dạng "Tên môn học (Mã môn học)" nếu tra được tên.
    Nếu mã không tồn tại trong map thì giữ nguyên mã để tránh bịa tên.
    """
    code = normalize_space(code)
    name = code_name_map.get(code)
    return f"{name} ({code})" if name else code


def list_course_labels(codes: list[str], code_name_map: dict[str, str]) -> str:
    return ", ".join(course_label(code, code_name_map) for code in codes)


def major_label(code: str, major_code_name_map: dict[str, str]) -> str:
    """
    Trả về dạng "Tên ngành/đơn vị (Mã)" nếu có trong major_code_name_map.
    Nếu không có thì giữ nguyên mã.
    """
    code = normalize_space(code)
    name = major_code_name_map.get(code)
    return f"{name} ({code})" if name else code


def unit_label_text(unit_text: str, major_code_name_map: dict[str, str]) -> str:
    unit_codes = split_unit_codes(unit_text)
    if not unit_codes:
        return ""

    return ", ".join(major_label(code, major_code_name_map) for code in unit_codes)


def build_content(
    row: dict[str, Any],
    code_name_map: dict[str, str],
    major_code_name_map: dict[str, str],
) -> str:
    current_course = course_label(row["ma_mon_hoc"], code_name_map)

    parts = [
        f"Môn học {current_course} có tên tiếng Anh là {row['ten_tieng_anh']}."
        if row["ten_tieng_anh"]
        else f"Môn học {current_course}.",
    ]

    if row["con_mo_lop"] is True:
        parts.append("Môn học này hiện đang mở lớp.")
    elif row["con_mo_lop"] is False:
        parts.append("Môn học này hiện không còn mở lớp.")

    unit_label = unit_label_text(row["don_vi_quan_ly_chuyen_mon"], major_code_name_map)
    if unit_label:
        parts.append(f"Môn học thuộc đơn vị quản lý chuyên môn {unit_label}.")

    if row["loai_mon_hoc"]:
        parts.append(f"Loại môn học là {row['loai_mon_hoc']}.")

    if row["ma_cu"]:
        parts.append(
            "Mã môn học cũ: "
            + list_course_labels(row["ma_cu"], code_name_map)
            + "."
        )

    if row["ma_mon_hoc_tuong_duong"]:
        parts.append(
            "Các môn học tương đương: "
            + list_course_labels(row["ma_mon_hoc_tuong_duong"], code_name_map)
            + "."
        )

    if row["ma_mon_hoc_tien_quyet"]:
        parts.append(
            "Các môn học tiên quyết: "
            + list_course_labels(row["ma_mon_hoc_tien_quyet"], code_name_map)
            + "."
        )

    if row["ma_mon_hoc_truoc"]:
        parts.append(
            "Các môn học trước: "
            + list_course_labels(row["ma_mon_hoc_truoc"], code_name_map)
            + "."
        )

    lt = row["so_tin_chi_ly_thuyet"]
    th = row["so_tin_chi_thuc_hanh"]

    if lt is not None or th is not None:
        lt_text = lt if lt is not None else 0
        th_text = th if th is not None else 0
        parts.append(
            f"Số tín chỉ lý thuyết là {lt_text}, số tín chỉ thực hành là {th_text}."
        )

    return " ".join(parts)


def parse_raw_courses(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = find_course_table(soup)

    rows = table.find_all("tr")
    raw_rows: list[dict[str, Any]] = []

    for tr in rows[1:]:
        cells = tr.find_all(["td", "th"])

        # Expected:
        # 0 Số TT
        # 1 Mã MH
        # 2 Tên MH tiếng Việt
        # 3 Tên MH tiếng Anh
        # 4 Còn mở lớp
        # 5 Đơn vị quản lý chuyên môn
        # 6 Loại MH
        # 7 Mã cũ
        # 8 Mã môn học tương đương
        # 9 Mã môn học tiên quyết
        # 10 Mã môn học trước
        # 11 Số TCLT
        # 12 Số TCTH
        if len(cells) < 13:
            continue

        stt = parse_int(clean_cell_text(cells[0]))
        course_code = normalize_space(clean_cell_text(cells[1])).replace("\n", " ")
        vietnamese_name = normalize_space(clean_cell_text(cells[2])).replace("\n", " ")
        english_name = normalize_space(clean_cell_text(cells[3])).replace("\n", " ")

        if not course_code or not vietnamese_name:
            continue

        status_text = clean_cell_text(cells[4])

        row = {
            "so_thu_tu": stt,
            "ma_mon_hoc": course_code,
            "ten_tieng_viet": vietnamese_name,
            "ten_tieng_anh": english_name,
            "trang_thai_mo_lop": normalize_space(status_text),
            "con_mo_lop": is_open_status(status_text),
            "don_vi_quan_ly_chuyen_mon": normalize_space(clean_cell_text(cells[5])),
            "loai_mon_hoc": normalize_space(clean_cell_text(cells[6])),
            "ma_cu": split_codes(clean_cell_text(cells[7])),
            "ma_mon_hoc_tuong_duong": split_codes(clean_cell_text(cells[8])),
            "ma_mon_hoc_tien_quyet": split_codes(clean_cell_text(cells[9])),
            "ma_mon_hoc_truoc": split_codes(clean_cell_text(cells[10])),
            "so_tin_chi_ly_thuyet": parse_int(clean_cell_text(cells[11])),
            "so_tin_chi_thuc_hanh": parse_int(clean_cell_text(cells[12])),
        }

        raw_rows.append(row)

    return raw_rows


def build_outputs(
    raw_rows: list[dict[str, Any]],
    major_code_name_map: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    # X:Y với X là mã môn học, Y là tên tiếng Việt.
    code_name_map = {
        row["ma_mon_hoc"]: row["ten_tieng_viet"]
        for row in raw_rows
    }

    records = []
    for idx, row in enumerate(raw_rows, start=1):
        content = build_content(row, code_name_map, major_code_name_map)

        unit_codes = split_unit_codes(row["don_vi_quan_ly_chuyen_mon"])
        unit_labels = [major_label(code, major_code_name_map) for code in unit_codes]

        record = {
            "id": idx,
            "link": SOURCE_URL,
            "parent_document": PARENT_DOCUMENT,
            "content": content,
            "metadata": {
                "category": "mon hoc",
                "majors": unit_codes,
                "major_labels": unit_labels,
                "year": 2026,
                "program": None,
            },
        }

        records.append(record)

    return records, code_name_map


def parse_courses(
    html: str,
    major_code_name_map: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    raw_rows = parse_raw_courses(html)
    return build_outputs(raw_rows, major_code_name_map)


def download_html() -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
    }

    response = requests.get(SOURCE_URL, headers=headers, timeout=60)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-html",
        default="",
        help="Nếu đã tải HTML sẵn thì truyền path HTML vào đây. Nếu bỏ trống, script tự tải từ DAA.",
    )
    parser.add_argument(
        "--output-dir",
        default="daa_course_outputs",
        help="Thư mục output.",
    )
    parser.add_argument(
        "--major-map",
        default=DEFAULT_MAJOR_MAP_PATH,
        help="File JSON map mã ngành/đơn vị sang tên đầy đủ, dạng {\"KHMT\": \"Khoa học máy tính\"}.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    major_code_name_map = load_json_map(args.major_map)

    if args.input_html:
        html = Path(args.input_html).read_text(encoding="utf-8")
    else:
        html = download_html()
        (output_dir / "danh_muc_mon_hoc_dai_hoc.html").write_text(html, encoding="utf-8")

    records, code_name_map = parse_courses(html, major_code_name_map)

    records_path = output_dir / "daa_courses_rag.json"
    map_path = output_dir / "course_code_name_map.json"
    major_map_output_path = output_dir / "major_code_name_map.json"

    records_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    map_path.write_text(
        json.dumps(code_name_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    major_map_output_path.write_text(
        json.dumps(major_code_name_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(records)} course records to {records_path}")
    print(f"Saved {len(code_name_map)} code-name mappings to {map_path}")
    print(f"Saved {len(major_code_name_map)} major/unit mappings to {major_map_output_path}")


if __name__ == "__main__":
    main()
