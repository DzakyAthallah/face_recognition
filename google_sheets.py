# google_sheets.py
import gspread
from google.oauth2.service_account import Credentials

class GoogleSheetsClient:
    def __init__(self, credentials_json='credentials.json', sheet_key=None, sheet_name='Sheet1'):
        self.sheet=None
        try:
            creds = Credentials.from_service_account_file(credentials_json, scopes=['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive'])
            self.client = gspread.authorize(creds)
            if sheet_key:
                ss = self.client.open_by_key(sheet_key)
            else:
                raise ValueError('sheet_key required')
            self.sheet = ss.worksheet(sheet_name)
        except Exception as e:
            print('GoogleSheets init error', e)
            self.sheet=None

    def ensure_header(self, headers=None):
        if self.sheet is None:
            return False
        if headers is None:
            headers=['Nama','Tanggal','Waktu','Confidence (%)']
        first_row = self.sheet.row_values(1)
        if not first_row or len(first_row)<len(headers):
            self.sheet.insert_row(headers,1)
        return True

    def record_attendance(self, name, extra=None):
        if self.sheet is None:
            return False
        from datetime import datetime
        ts = datetime.now()
        tanggal = ts.strftime('%Y-%m-%d')
        waktu = ts.strftime('%H:%M:%S')
        confidence = extra if extra is not None else '-'
        try:
            self.sheet.append_row([name, tanggal, waktu, confidence], value_input_option='USER_ENTERED')
            return True
        except Exception as e:
            print('Sheet write error', e)
            return False
