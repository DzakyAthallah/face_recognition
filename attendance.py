# attendance.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

class GoogleSheetsClient:
    def __init__(self, sheet_name="Absensi Face Recognition"):
        """
        Inisialisasi koneksi Google Sheets
        """
        try:
            scope = [
                "https://docs.google.com/spreadsheets/d/1KEIGm7k59yE3iw3tWDHFVT0t_9V9bAj1eBNlkKp90oM/edit?gid=0#gid=0",
                "https://drive.google.com/drive/folders/1-iDI_pp5wXXxRYuTHeCbmRRY6Syf6R7N?usp=sharing"
            ]

            creds = ServiceAccountCredentials.from_json_keyfile_name(
                "credentials.json", scope
            )

            client = gspread.authorize(creds)

            # Buka sheet jika ada, kalau tidak buat baru
            try:
                self.sheet = client.open(sheet_name).sheet1
            except gspread.SpreadsheetNotFound:
                sh = client.create(sheet_name)
                sh.share(None, perm_type='anyone', role='writer')
                self.sheet = sh.sheet1

            # Set header jika masih kosong
            if self.sheet.row_count <= 1 and not self.sheet.get_all_values():
                self.sheet.append_row(["Nama", "Tanggal", "Waktu", "Confidence (%)"])
                print("📄 Header spreadsheet dibuat!")

        except Exception as e:
            print("❌ ERROR Google Sheets:", e)
            self.sheet = None

    def record_attendance(self, name, extra=None):
        """
        extra akan diisi confidence (%)
        """
        if self.sheet is None:
            print("⚠️ Google Sheet tidak tersedia.")
            return False

        now = datetime.now()
        tanggal = now.strftime("%Y-%m-%d")
        waktu = now.strftime("%H:%M:%S")

        confidence = extra if extra is not None else "-"

        try:
            self.sheet.append_row([name, tanggal, waktu, confidence])
            print(f"📌 Absensi tersimpan ke Google Sheets: {name}")
            return True
        except Exception as e:
            print("❌ Gagal menyimpan ke Google Sheets:", e)
            return False
