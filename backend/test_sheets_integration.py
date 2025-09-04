import os
import traceback

# Ensure the test uses the same explicit credentials path as .env
os.environ['GOOGLE_SHEETS_CREDENTIALS'] = r"C:\Users\PC\AppData\Roaming\gspread\service_account.json"

try:
    from google_sheets_integration import sheets_logger
except Exception as e:
    print('FAILED importing google_sheets_integration:', e)
    traceback.print_exc()
    raise

try:
    print('Sheets client present:', hasattr(sheets_logger, 'client') and sheets_logger.client is not None)
    # Setup or open a test spreadsheet
    sheets_logger.setup_spreadsheet(spreadsheet_name='Test WhatsApp AI Logs')
    ss = getattr(sheets_logger, 'spreadsheet', None)
    if ss:
        print('Spreadsheet title:', ss.title)
        try:
            sid = ss.id
            print('Spreadsheet id:', sid)
            print('Spreadsheet url: https://docs.google.com/spreadsheets/d/' + sid)
        except Exception:
            pass
    else:
        print('No spreadsheet object available; client initialized:', getattr(sheets_logger, 'client', None) is not None)

    # Append a test conversation summary
    test_summary = {
        'phone_number': '+1234567890',
        'contact_name': 'Test User',
        'is_known_contact': False,
        'message_count': 1,
        'priority': 'low',
        'category': 'test',
        'summary': 'Automated test entry from local env',
        'duration_seconds': 0,
        'first_message': 'hello',
        'last_message': 'hello',
        'status': 'test'
    }

    sheets_logger.log_conversation_summary(test_summary)
    print('Appended test summary (check spreadsheet or service account Drive).')

except Exception as e:
    print('ERROR during Sheets test:', e)
    traceback.print_exc()
    raise
