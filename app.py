import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="MWA Base File Generator", layout="wide")

st.title("MWA Base File Generator")
st.write(
    "Upload the **FA Export file** and the **Manpower file** to generate the "
    "Base File with the 14 additional columns required for the daily MWA report."
)

SHEET_NAME = "Visit Data Upload 2023-nerolive"

NEW_COLUMNS_ORDER = [
    "User Group",
    "Sales Group",
    "SG Flag",
    "DOJ",
    "DOJ Month",
    "Business",
    "Visit Month",
    "Business Partner",
    "Business Category",
    "Depot Code",
    "Division",
    "Zone",
    "Officer",
    "Parent customer",
]

# Manpower file column names used for lookups
# Join key: FA export's KNPL_ID <-> Manpower file's "Emp ID"
MANPOWER_LOOKUP_KEY = "Emp ID"
FA_LOOKUP_KEY = "KNPL_ID"
MANPOWER_COLS_NEEDED = {
    "Sales Group": "Sales Group",
    "SG Flag": "SG FLAG",
    "DOJ": "DOJ",
    "Business": "Business",
    "Depot Code": "Depot Code",
    "Division": "Division",
    "Zone": "Zone.1",
    "Officer": "Officers",
}


def read_any(file):
    """Read an uploaded file (xlsx or csv) into a DataFrame, all columns as string.
    Uses the 'calamine' engine for Excel files - significantly faster than
    openpyxl for large files."""
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, dtype=str, keep_default_na=False)
    else:
        try:
            return pd.read_excel(file, dtype=str, keep_default_na=False, engine="calamine")
        except Exception:
            # Fallback to openpyxl if calamine is unavailable or fails (e.g. older .xls files)
            file.seek(0)
            return pd.read_excel(file, dtype=str, keep_default_na=False)


def strip_leading_zeros_and_prefix_column(series: pd.Series) -> pd.Series:
    """Vectorized version of strip_leading_zeros_and_prefix for an entire column."""
    s = series.astype(str).str.strip()
    empty_mask = s.eq("") | s.str.lower().isin(["nan", "none"])
    s = s.where(~empty_mask, "0")

    # Remove float artifacts like "12345.0" -> "12345"
    s = s.str.replace(r"\.0$", "", regex=True)

    # Handle scientific notation rows individually (rare)
    sci_mask = s.str.contains("e", case=False, na=False)
    if sci_mask.any():
        def _sci_to_int_str(x):
            try:
                return str(int(float(x)))
            except (ValueError, OverflowError):
                return x
        s.loc[sci_mask] = s.loc[sci_mask].apply(_sci_to_int_str)

    stripped = s.str.lstrip("0")
    result = "0000" + stripped
    result = result.where(stripped != "", "0000")
    return result


def format_date_value(value):
    """Convert a date-like value to DD.MM.YYYY text format.
    Handles datetime objects, pandas Timestamps, and strings like
    '2018-07-23 00:00:00' or '23-07-2018'."""
    if value is None:
        return ""
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "nat"):
        return ""

    import re

    # ISO format YYYY-MM-DD (with optional time) is unambiguous - handle first
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        parsed = pd.to_datetime(s, errors="coerce", format=None)
        if pd.notna(parsed):
            return parsed.strftime("%d.%m.%Y")

    # Try parsing as a datetime - prioritize day-first (DD-MM-YYYY) format
    # since this is the standard convention used in the source data
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.notna(parsed):
        return parsed.strftime("%d.%m.%Y")

    # Fallback: try month-first parsing
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.notna(parsed):
        return parsed.strftime("%d.%m.%Y")

    return s


def format_date_column(series: pd.Series) -> pd.Series:
    """Vectorized version of format_date_value for an entire column.
    Much faster than .apply(format_date_value) row-by-row for large files."""
    s = series.astype(str).str.strip()
    empty_mask = s.eq("") | s.str.lower().isin(["nan", "none", "nat"])

    # ISO format YYYY-MM-DD (with optional time) - unambiguous
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}")

    result = pd.Series("", index=series.index, dtype=object)

    # Parse ISO-format values (no dayfirst needed - unambiguous)
    if iso_mask.any():
        parsed_iso = pd.to_datetime(s[iso_mask], errors="coerce")
        result.loc[iso_mask] = parsed_iso.dt.strftime("%d.%m.%Y")

    # Parse remaining non-empty, non-ISO values with dayfirst=True
    remaining_mask = (~iso_mask) & (~empty_mask)
    if remaining_mask.any():
        parsed_day = pd.to_datetime(s[remaining_mask], errors="coerce", dayfirst=True)
        result.loc[remaining_mask] = parsed_day.dt.strftime("%d.%m.%Y")

        # Fallback to dayfirst=False for any that still failed
        still_failed = remaining_mask & result.eq("") & result.notna()
        still_failed = remaining_mask & (result == "")
        if still_failed.any():
            parsed_month = pd.to_datetime(s[still_failed], errors="coerce", dayfirst=False)
            result.loc[still_failed] = parsed_month.dt.strftime("%d.%m.%Y")

    # Anything still empty/NaN -> fall back to original string (or "")
    still_empty = result.isna() | result.eq("")
    fallback_mask = still_empty & (~empty_mask)
    result.loc[fallback_mask] = s.loc[fallback_mask]
    result.loc[empty_mask] = ""

    return result.fillna("")


def extract_month_only_column(series: pd.Series) -> pd.Series:
    """Extract month (MM) from a formatted date string like DD.MM.YYYY."""
    # Force to plain python strings first so split() never produces a
    # numeric/NaN (float64) column when a value doesn't have the expected
    # number of '.'-separated parts.
    s = series.astype(str).str.strip()
    empty_mask = s.eq("") | s.str.lower().isin(["nan", "none"])

    parts = s.str.split(".")

    def _get_month(p):
        if isinstance(p, list) and len(p) >= 2:
            return p[1]
        return ""

    result = parts.apply(_get_month)
    result = result.astype(str)
    result.loc[empty_mask] = ""
    return result


def extract_month_year_column(series: pd.Series) -> pd.Series:
    """Extract MM.YYYY from a formatted date string like DD.MM.YYYY."""
    # NOTE: previously this used parts.str[1] + "." + parts.str[2], which
    # returns NaN (and can silently degrade the whole column to float64)
    # whenever a row doesn't split into at least 3 parts - e.g. an empty
    # string, or a date that format_date_column couldn't parse and left
    # as raw/partial text. Concatenating "." (a str) with a float64 NaN
    # column then raises:
    #   ufunc 'add' did not contain a loop with signature matching types
    #   (dtype('float64'), dtype('<U1')) -> None
    # Building the result row-by-row with plain Python avoids that.
    s = series.astype(str).str.strip()
    empty_mask = s.eq("") | s.str.lower().isin(["nan", "none"])

    parts = s.str.split(".")

    def _get_month_year(p):
        if isinstance(p, list) and len(p) >= 3:
            return f"{p[1]}.{p[2]}"
        return ""

    result = parts.apply(_get_month_year)
    result = result.astype(str)
    result.loc[empty_mask] = ""
    return result


def build_base_file(fa_df: pd.DataFrame, manpower_df: pd.DataFrame) -> pd.DataFrame:
    fa = fa_df.copy()
    mp = manpower_df.copy()

    # Drop any pre-existing placeholder versions of the 14 new columns
    # (some FA export files come with these columns already present but empty)
    fa = fa.drop(columns=[c for c in NEW_COLUMNS_ORDER if c in fa.columns], errors="ignore")

    # Validate required columns
    required_fa_cols = [
        "SGRP",
        "USER_NAME",
        "ACTUAL_DATE",
        "BUSINESS_PARTNER_ROLE",
        "PROSPECT_CUST_ID",
        FA_LOOKUP_KEY,
    ]
    missing_fa = [c for c in required_fa_cols if c not in fa.columns]
    if missing_fa:
        raise ValueError(f"FA Export file is missing required columns: {missing_fa}")

    required_mp_cols = [MANPOWER_LOOKUP_KEY] + [MANPOWER_COLS_NEEDED[k] for k in MANPOWER_COLS_NEEDED]
    missing_mp = [c for c in required_mp_cols if c not in mp.columns]
    if missing_mp:
        raise ValueError(f"Manpower file is missing required columns: {missing_mp}")

    original_fa_columns = list(fa.columns)

    # --- Build lookup table from Manpower file, keyed on "Emp ID" ---
    lookup_cols = [MANPOWER_LOOKUP_KEY] + [MANPOWER_COLS_NEEDED[k] for k in MANPOWER_COLS_NEEDED]
    lookup_df = mp[lookup_cols].drop_duplicates(subset=[MANPOWER_LOOKUP_KEY], keep="first")

    # Rename manpower columns to the new base-file column names for the merge
    rename_map = {MANPOWER_COLS_NEEDED[k]: k for k in MANPOWER_COLS_NEEDED}
    lookup_df = lookup_df.rename(columns=rename_map)

    # --- Merge (left join) FA export with lookup table on KNPL_ID == Emp ID ---
    merged = fa.merge(
        lookup_df,
        how="left",
        left_on=FA_LOOKUP_KEY,
        right_on=MANPOWER_LOOKUP_KEY,
    )

    # --- Format date columns from FA Export if they exist ---
    date_columns_to_format = ["CREATION_DATE", "SYSTEM_DAY", "DATE_OF_ENTRY_IN_TABLE", "UPDATEDDATE"]
    for col in date_columns_to_format:
        if col in merged.columns:
            merged[col] = format_date_column(merged[col])

    # --- Derived columns (no lookup) ---
    merged["User Group"] = merged["USER_NAME"].astype(str).str[:3]
    merged["DOJ"] = format_date_column(merged["DOJ"])
    merged["DOJ Month"] = extract_month_year_column(merged["DOJ"])  # MM.YYYY
    merged["Visit Month"] = extract_month_year_column(format_date_column(merged["ACTUAL_DATE"]))  # MM.YYYY
    merged["Business Partner"] = merged["BUSINESS_PARTNER_ROLE"]
    bp_blank_mask = merged["Business Partner"].isna() | (merged["Business Partner"].astype(str).str.strip() == "")
    merged.loc[bp_blank_mask, "Business Partner"] = "0"
    merged["Business Category"] = ""
    merged["Parent customer"] = strip_leading_zeros_and_prefix_column(merged["PROSPECT_CUST_ID"])

    looked_up_cols = ["Sales Group", "SG Flag", "DOJ", "Business", "Depot Code", "Division", "Zone", "Officer"]

    # --- Cleanup Rule: literal "N.A" in looked-up Sales Group -> "Not Assigned" ---
    na_literal_mask = merged["Sales Group"].astype(str).str.strip().str.upper() == "N.A"
    merged.loc[na_literal_mask, "Sales Group"] = "Not Assigned"

    # --- Cleanup Rule: failed lookup (no match in Manpower file via KNPL_ID == Emp ID) ---
    # If KNPL_ID has no corresponding Emp ID in the Manpower file, set Sales Group, Business,
    # and all other looked-up columns (including DOJ Month) to "Not Assigned".
    lookup_failed_mask = merged[MANPOWER_LOOKUP_KEY].isna() | (merged[MANPOWER_LOOKUP_KEY].astype(str).str.strip() == "")
    merged.loc[lookup_failed_mask, "Sales Group"] = "Not Assigned"
    merged.loc[lookup_failed_mask, "Business"] = "Not Assigned"
    for col in ["SG Flag", "DOJ", "DOJ Month", "Depot Code", "Division", "Zone", "Officer"]:
        merged.loc[lookup_failed_mask, col] = "Not Assigned"

    # --- Final assembly: keep only original FA columns + 14 new columns, in order ---
    final_columns = original_fa_columns + NEW_COLUMNS_ORDER
    result = merged[final_columns].copy()

    # --- Cleanup Rule C: final sweep for #NA / #N/A / NaN across ALL columns ---
    # --- Convert everything to text/string in one pass, avoiding float artifacts ---
    result = result.astype(str)
    result = result.replace(
        to_replace=["nan", "NaN", "None", "<NA>", "#NA", "#N/A", "NAN"],
        value="Not Assigned",
    )

    return result


# --- Streamlit UI ---
tab1, tab2 = st.tabs(["Generate Base File", "Compare Files"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fa_file = st.file_uploader("FA Export file (Visit Data Upload)", type=["xlsx", "xls", "csv"], key="fa_file")
    with col2:
        manpower_file = st.file_uploader("Manpower file", type=["xlsx", "xls", "csv"], key="manpower_file")

    if st.button("Process and Generate Base File", type="primary"):
        if fa_file is None or manpower_file is None:
            st.error("Please upload both files before processing.")
        else:
            try:
                with st.spinner("Reading files..."):
                    fa_df = read_any(fa_file)
                    manpower_df = read_any(manpower_file)

                with st.spinner("Processing base file..."):
                    result_df = build_base_file(fa_df, manpower_df)

                st.success(f"Base file generated successfully — {len(result_df)} rows, {len(result_df.columns)} columns.")

                # Write to Excel in memory (xlsxwriter is faster than openpyxl for large files)
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    result_df.to_excel(writer, sheet_name=SHEET_NAME, index=False)
                    # Hidden document metadata (not visible in any cell/UI)
                    writer.book.set_properties({
                        "author": "Allen Peter",
                        "last_modified_by": "Allen Peter",
                    })

                excel_bytes = output.getvalue()

                st.download_button(
                    label="⬇️ Download Base File (.xlsx)",
                    data=excel_bytes,
                    file_name="Base_File.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                )

                st.caption(
                    "Preview (first 50 rows) — use the green button above to download the full .xlsx file. "
                    "The download icon on the table below exports as CSV, not the Base File."
                )
                st.dataframe(result_df.head(50), use_container_width=True)
            except Exception as e:
                st.error(f"Error while processing files: {e}")

with tab2:
    st.write(
        "Upload the **tool-generated Base File** and a **manually-prepared file** to compare them "
        "row-by-row (in order) and cell-by-cell. The report will list every cell where the two "
        "files differ."
    )

    col1, col2 = st.columns(2)
    with col1:
        generated_file = st.file_uploader("Tool-generated file", type=["xlsx", "xls", "csv"], key="generated_file")
    with col2:
        manual_file = st.file_uploader("Manually-prepared file", type=["xlsx", "xls", "csv"], key="manual_file")

    if st.button("Compare Files", type="primary"):
        if generated_file is None or manual_file is None:
            st.error("Please upload both files before comparing.")
        else:
            try:
                with st.spinner("Reading files..."):
                    gen_df = read_any(generated_file)
                    man_df = read_any(manual_file)

                with st.spinner("Comparing..."):
                    diffs = []

                    # Compare row counts
                    n_rows = min(len(gen_df), len(man_df))
                    if len(gen_df) != len(man_df):
                        st.warning(
                            f"Row count mismatch: tool-generated file has {len(gen_df)} rows, "
                            f"manual file has {len(man_df)} rows. Comparing only the first {n_rows} rows."
                        )

                    # Compare column sets
                    gen_cols = list(gen_df.columns)
                    man_cols = list(man_df.columns)
                    common_cols = [c for c in gen_cols if c in man_cols]
                    only_in_gen = [c for c in gen_cols if c not in man_cols]
                    only_in_man = [c for c in man_cols if c not in gen_cols]

                    if only_in_gen:
                        st.info(f"Columns only in tool-generated file: {only_in_gen}")
                    if only_in_man:
                        st.info(f"Columns only in manual file: {only_in_man}")

                    # Align on common columns, first n_rows
                    gen_aligned = gen_df[common_cols].head(n_rows).reset_index(drop=True).astype(str)
                    man_aligned = man_df[common_cols].head(n_rows).reset_index(drop=True).astype(str)

                    # Normalize NaN-like strings for fair comparison
                    gen_aligned = gen_aligned.replace(["nan", "None", "<NA>"], "")
                    man_aligned = man_aligned.replace(["nan", "None", "<NA>"], "")

                    for col in common_cols:
                        mismatch_mask = gen_aligned[col] != man_aligned[col]
                        if mismatch_mask.any():
                            mismatched_idx = gen_aligned.index[mismatch_mask]
                            for idx in mismatched_idx:
                                diffs.append({
                                    "Row (Excel row, 1-indexed incl. header)": idx + 2,
                                    "Column": col,
                                    "Tool-generated Value": gen_aligned.at[idx, col],
                                    "Manual File Value": man_aligned.at[idx, col],
                                })

                if not diffs:
                    st.success(f"✅ No differences found across {n_rows} rows and {len(common_cols)} common columns.")
                else:
                    diff_df = pd.DataFrame(diffs)
                    st.error(f"Found {len(diff_df)} differing cells across {n_rows} rows.")
                    st.dataframe(diff_df, use_container_width=True)

                    # Provide downloadable diff report
                    diff_output = BytesIO()
                    with pd.ExcelWriter(diff_output, engine="xlsxwriter") as writer:
                        diff_df.to_excel(writer, sheet_name="Differences", index=False)
                        writer.book.set_properties({
                            "author": "Allen Peter",
                            "last_modified_by": "Allen Peter",
                        })
                    diff_bytes = diff_output.getvalue()

                    st.download_button(
                        label="⬇️ Download Differences Report (.xlsx)",
                        data=diff_bytes,
                        file_name="Differences_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )
            except Exception as e:
                st.error(f"Error while comparing files: {e}")
