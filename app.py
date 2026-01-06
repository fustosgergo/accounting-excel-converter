# app.py
import re
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st


def clean_df(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    idx = df[~pd.to_numeric(df[col_name], errors="coerce").notna()].index.min()
    return df.loc[: idx - 1] if pd.notna(idx) else df


def parse_filename_stem(stem: str) -> tuple[int, str]:
    m = re.match(r"^(\d+)_([0-9]{8})$", stem)
    if not m:
        raise ValueError(f"Unexpected filename format: {stem}")
    return int(m.group(1)), m.group(2)


def build_output(
    src: pd.DataFrame,
    output_cols: list[str],
    datum: str,
    biz: str,
    mode: str,  # "vevo" | "szall"
) -> pd.DataFrame:
    if mode == "vevo":
        amount_col = "Érték"

        def tk_rule(x):
            return "T" if x > 0 else "K"

    elif mode == "szall":
        amount_col = "Összeg"

        def tk_rule(x):
            return "K" if x > 0 else "T"

    else:
        raise ValueError("mode must be 'vevo' or 'szall'")

    out = pd.DataFrame("", index=range(len(src)), columns=output_cols)
    return out.assign(
        naplo="B",
        Kelt=datum,
        Biz=biz,
        Fmod=2,
        Kiegybiz=src["Számlaszám"],
        Brt=src[amount_col].abs(),
        Bfok=3841,
        Btk=src[amount_col].apply(tk_rule),
    )


@st.cache_data
def load_output_cols() -> list[str]:
    return pd.read_csv("output_cols.csv", nrows=0, sep=";").columns.tolist()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=False, sep=";", encoding="cp1252")
    return buffer.getvalue()


def process_one_xlsx(
    file_bytes: bytes, original_name: str, output_cols: list[str]
) -> dict[str, bytes]:
    sorszam, _ = parse_filename_stem(Path(original_name).stem)

    dfs = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
    vevo_df = clean_df(dfs["Vevő HUF"], col_name="Érték")
    szall_df = clean_df(dfs["Szállító HUF"], col_name="Összeg")

    datum = vevo_df["Keltezés"].iloc[0]
    out_filename = "".join(datum.split(".")[1:])
    biz = f"{sorszam}/{datum.split('.')[0]}"

    vevo_output = build_output(vevo_df, output_cols, datum, biz, mode="vevo")
    szall_output = build_output(szall_df, output_cols, datum, biz, mode="szall")

    return {
        f"{out_filename}_vevo.csv": df_to_csv_bytes(vevo_output),
        f"{out_filename}_szall.csv": df_to_csv_bytes(szall_output),
    }


def main() -> None:
    st.set_page_config(page_title="Excel → CSV", layout="centered")
    st.title("Excel → CSV")

    output_cols = load_output_cols()

    uploaded = st.file_uploader(
        "Tölts fel egy vagy több .xlsx fájlt (pl. 0203_20250801.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    run = st.button("Konvertálás", type="primary", disabled=not uploaded)

    if run:
        zip_buf = BytesIO()
        results = []

        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for f in uploaded:
                try:
                    out_files = process_one_xlsx(f.getvalue(), f.name, output_cols)

                    for out_name, out_bytes in out_files.items():
                        z.writestr(out_name, out_bytes)  # pl. 0801_vevo.csv
                    results.append((f.name, "OK"))
                except Exception as e:
                    results.append((f.name, f"Hiba: {e}"))

        st.subheader("Eredmény")
        st.table(pd.DataFrame(results, columns=["Fájl", "Státusz"]))

        zip_buf.seek(0)
        st.download_button(
            "Letöltés ZIP-ben",
            data=zip_buf.getvalue(),
            file_name="converted.zip",
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
