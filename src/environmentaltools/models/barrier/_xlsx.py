"""Excel tables of the BARRIER workflow: Arial, bold frozen header, filters, fitted widths and a legend sheet."""

import pandas as pd

CHANGED_FILL = "F8CBAD"   # orange: value different from the base case


def write_tables(path, sheets, legend=None, highlight=None):
    """Write DataFrames as sheets of one Excel file with a ``legend`` sheet.

    Args:
        path (str | Path): Output file.
        sheets (dict): ``{sheet name: DataFrame}`` (the index is not written).
        legend (dict | None): ``{column or item: description}``, written in the ``legend`` sheet.
        highlight (dict | None): ``{sheet name: (reference column, first compared column)}``: cells of
            the compared columns different from the reference column are filled in orange.

    Returns:
        Path: The written file.
    """
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
        if legend:
            pd.DataFrame({"item": list(legend), "description": list(legend.values())}).to_excel(
                xw, sheet_name="legend", index=False)
        for ws in xw.book.worksheets:
            for row in ws.iter_rows():
                for c in row:
                    c.font = Font(name="Arial", bold=c.row == 1)
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for i, col in enumerate(ws.iter_cols(), start=1):
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 10), 60)
        if legend:
            ws = xw.book["legend"]
            for c in ws["B"]:
                c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.column_dimensions["B"].width = 100
        fill = PatternFill("solid", start_color=CHANGED_FILL)
        for name, (ref_col, first_col) in (highlight or {}).items():
            df = sheets[name]
            r, f = df.columns.get_loc(ref_col), df.columns.get_loc(first_col)
            ws = xw.book[name]
            for i, row in enumerate(df.itertuples(index=False), start=2):
                for j in range(f, len(df.columns)):
                    if str(row[j]) != str(row[r]):
                        ws.cell(i, j + 1).fill = fill
    return path
