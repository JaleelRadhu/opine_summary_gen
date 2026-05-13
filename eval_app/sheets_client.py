"""Google Sheets connection helper.

Credentials are read from st.secrets (Streamlit Community Cloud or local
.streamlit/secrets.toml). Falls back gracefully when Sheets is not configured
so that local CSV/JSON storage still works.
"""
import gspread
import streamlit as st
from google.oauth2.service_account import Credentials

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def sheets_configured() -> bool:
    try:
        return "gcp_service_account" in st.secrets and "SPREADSHEET_ID" in st.secrets
    except Exception:
        return False


@st.cache_resource
def _get_client() -> gspread.Client:
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=_SCOPES,
    )
    return gspread.authorize(creds)


def get_worksheet(sheet_name: str) -> gspread.Worksheet:
    gc = _get_client()
    return gc.open_by_key(st.secrets["SPREADSHEET_ID"]).worksheet(sheet_name)
