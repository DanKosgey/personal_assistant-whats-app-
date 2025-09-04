import os, traceback
os.environ['GOOGLE_SHEETS_CREDENTIALS'] = r"C:\Users\PC\AppData\Roaming\gspread\service_account.json"

try:
    from google_sheets_integration import sheets_logger
    print('sheets_logger.client:', getattr(sheets_logger, 'client', None))
    ss = getattr(sheets_logger, 'spreadsheet', None)
    if ss is None:
        try:
            sheets_logger.setup_spreadsheet(spreadsheet_name='Test WhatsApp AI Logs')
            ss = sheets_logger.spreadsheet
        except Exception as e:
            print('Error setting up/opening spreadsheet:', e)
            traceback.print_exc()
    if ss:
        print('Spreadsheet title:', ss.title)
        print('Spreadsheet id:', ss.id)
        try:
            ws = sheets_logger.worksheet
            vals = ws.get_all_values()
            print('Worksheet rows:', len(vals))
            if vals:
                print('Last row:', vals[-1])
        except Exception as e:
            print('Error reading worksheet:', e)
            traceback.print_exc()
    else:
        print('No spreadsheet available; client initialized:', getattr(sheets_logger, 'client', None) is not None)
except Exception as e:
    print('Failed to import or run:', e)
    traceback.print_exc()
