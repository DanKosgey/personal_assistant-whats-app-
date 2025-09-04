import os
os.environ['GOOGLE_SHEETS_CREDENTIALS'] = r"C:\Users\PC\AppData\Roaming\gspread\service_account.json"
from google_sheets_integration import sheets_logger
print('service_account_email=', getattr(sheets_logger, 'service_account_email', None))
print('spreadsheet_id=', getattr(sheets_logger, 'spreadsheet_id', None))
print('client exists=', getattr(sheets_logger, 'client', None) is not None)
