"""
Excel format inference and output.
Read training Excel, infer structure, write inference results in same format.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import openpyxl
    from openpyxl import load_workbook, Workbook
except ImportError:
    openpyxl = None


def load_excel_format(excel_path: str | Path) -> Dict[str, Any]:
    """
    Load Excel and infer structure (headers, column indices).
    Returns dict with: headers, sheet_name, sample_row.
    """
    if openpyxl is None:
        raise ImportError("openpyxl required: pip install openpyxl")
    wb = load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb.active
    headers = [str(c.value or "").strip() for c in ws[1]]
    sample = [c.value for c in ws[2]] if ws.max_row >= 2 else []
    wb.close()
    return {
        "headers": headers,
        "sheet_name": ws.title,
        "sample_row": sample,
    }


def infer_column_indices(headers: List[str]) -> Dict[str, int]:
    """
    Infer column indices from headers.
    Common names: 위핀, 아래핀, 위핀개수, 아래핀개수, OK, NG, 판정, 좌우간격, spacing, etc.
    """
    mapping = {}
    for i, h in enumerate(headers):
        h_lower = h.lower().replace(" ", "")
        if "위" in h and ("핀" in h or "개" in h):
            mapping["upper_count"] = i
        elif "아래" in h and ("핀" in h or "개" in h):
            mapping["lower_count"] = i
        elif "ok" in h_lower or "ng" in h_lower or "판정" in h:
            mapping["judgment"] = i
        elif "간격" in h or "spacing" in h_lower or "거리" in h:
            mapping["spacing"] = i
    return mapping


def write_result_excel(
    output_path: str | Path,
    upper_count: int,
    lower_count: int,
    upper_spacings: List[float],
    lower_spacings: List[float],
    format_ref: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write inference result to Excel.
    If format_ref from load_excel_format, use same structure; else minimal format.
    """
    if openpyxl is None:
        raise ImportError("openpyxl required: pip install openpyxl")
    wb = Workbook()
    ws = wb.active
    judgment = "OK" if (upper_count == 20 and lower_count == 20) else "NG"

    if format_ref and format_ref.get("headers"):
        headers = format_ref["headers"]
        ws.append(headers)
        row = [""] * len(headers)
        idx = infer_column_indices(headers)
        if "upper_count" in idx:
            row[idx["upper_count"]] = upper_count
        if "lower_count" in idx:
            row[idx["lower_count"]] = lower_count
        if "judgment" in idx:
            row[idx["judgment"]] = judgment
        if "spacing" in idx:
            row[idx["spacing"]] = ", ".join(f"{s:.2f}" for s in (upper_spacings + lower_spacings))
        ws.append(row)
    else:
        ws.append(["위핀개수", "아래핀개수", "판정", "위핀간격(mm)", "아래핀간격(mm)"])
        ws.append([
            upper_count,
            lower_count,
            judgment,
            ", ".join(f"{s:.2f}" for s in upper_spacings),
            ", ".join(f"{s:.2f}" for s in lower_spacings),
        ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
