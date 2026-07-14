from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any, Mapping


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, output)
    else:
        output[prefix] = value


def build_report(
    summary: Mapping[str, Any],
    output_dir: str | Path,
    *,
    visual_gallery: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_paths = []
    for section_key, title in (
        ("main_methods", "main_methods"),
        ("diagnostic_methods", "diagnostic_methods"),
    ):
        rows = []
        columns = {"method"}
        for method, payload in summary.get(section_key, {}).items():
            row = {"method": method}
            _flatten("", payload, row)
            rows.append(row)
            columns.update(row)
        ordered = ["method"] + sorted(columns - {"method"})
        path = output_dir / f"{title}.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=ordered)
            writer.writeheader()
            writer.writerows(rows)
        csv_paths.append(str(path))
    html_path = output_dir / "report.html"

    def render_section(key: str, title: str) -> str:
        methods = summary.get(key, {})
        rows = []
        for method, payload in methods.items():
            flat = {"method": method}
            _flatten("", payload, flat)
            rows.append(flat)
        columns = ["method"] + sorted(
            {col for row in rows for col in row if col != "method"}
        )
        header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
        body = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(col, '')))}</td>"
                for col in columns
            )
            + "</tr>"
            for row in rows
        )
        return (
            f"<h2>{html.escape(title)}</h2><table><thead><tr>{header}</tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

    visual_section = ""
    if visual_gallery is not None:
        gallery_path = Path(visual_gallery)
        if not gallery_path.is_file():
            raise FileNotFoundError(gallery_path)
        relative_gallery = Path(
            os.path.relpath(gallery_path, html_path.parent)
        ).as_posix()
        visual_section = (
            "<section><h2>Visual evidence</h2>"
            f"<p><a href='{html.escape(relative_gallery)}'>"
            "Open representative, improvement, failure and full-scene galleries"
            "</a></p></section>"
        )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>adjustTM Benchmark</title>"
        "<style>body{font-family:Arial,sans-serif;margin:32px}"
        "table{border-collapse:collapse;font-size:12px}"
        "th,td{border:1px solid #ccc;padding:6px}th{background:#eee}</style>"
        "<h1>adjustTM Benchmark Report</h1>"
        + visual_section
        + f"<pre>{html.escape(json.dumps(summary.get('protocol', {}), indent=2, ensure_ascii=False))}</pre>"
        + render_section("main_methods", "Main methods")
        + render_section("diagnostic_methods", "Diagnostic oracle methods"),
        encoding="utf-8",
    )
    outputs: dict[str, Any] = {
        "json": str(json_path),
        "html": str(html_path),
        "csv": csv_paths,
    }
    if visual_gallery is not None:
        outputs["visual_gallery"] = str(Path(visual_gallery))
    try:
        from openpyxl import Workbook

        workbook = Workbook()
        workbook.remove(workbook.active)
        for section_key, sheet_name in (
            ("main_methods", "Main Methods"),
            ("diagnostic_methods", "Oracles"),
        ):
            sheet = workbook.create_sheet(sheet_name)
            rows = []
            columns = {"method"}
            for method, payload in summary.get(section_key, {}).items():
                row = {"method": method}
                _flatten("", payload, row)
                rows.append(row)
                columns.update(row)
            ordered = ["method"] + sorted(columns - {"method"})
            sheet.append(ordered)
            for row in rows:
                sheet.append([row.get(column, "") for column in ordered])
            sheet.freeze_panes = "A2"
        xlsx_path = output_dir / "comparison.xlsx"
        workbook.save(xlsx_path)
        outputs["xlsx"] = str(xlsx_path)
    except ImportError:
        pass
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render adjustTM benchmark JSON summary to HTML and CSV"
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-gallery")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    print(
        json.dumps(
            build_report(
                summary,
                args.output_dir,
                visual_gallery=args.visual_gallery,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
