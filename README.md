# WhatsApp AI Agent

Detailed README for the Advanced WhatsApp AI Agent (backend)

This repository contains a production-capable WhatsApp AI agent that:
- Receives WhatsApp webhook messages and processes them.
- Uses multiple AI backends (Gemini primary, OpenRouter fallback) with key rotation.
- Applies ML-assisted analysis (sentiment, intent, priority, category).
- Sends WhatsApp replies with safety (allowlist, retries, backoff) and observability.
- Supports runtime persona switching (PersonaManager) so the agent's role can be changed by editing a system prompt.

This README explains architecture, setup, environment variables, persona usage, running locally, testing, deployment notes, and troubleshooting.

## Table of contents
- Overview
- Architecture
- Quickstart (local)
- Environment variables (.env)
- Persona management (swap agent by prompt)
- How AI fallback & key rotation work
- Running (development & production)
- Testing and simulation
- Observability & logs
- Security & secrets
- Troubleshooting (common errors)
- Scaling & production recommendations
- Contributing
- License

## Overview

The WhatsApp AI Agent is designed to automate conversational handling on WhatsApp for use cases such as personal assistance, customer support, lead capture, and more.

Key features:
- Conversation-aware replies with context preservation
- Multi-key rotation and graceful fallback from Google Gemini -> OpenRouter
- Persona system: change behavior by swapping a system prompt
- Production-ready WhatsApp sending with allowlist, retries, exponential backoff
- Database persistence (MongoDB) and optional Redis cache

The main backend code lives in `agent/whats-app-agent/backend/` and exposes a FastAPI application with webhook endpoints and admin routes.

## Architecture

- FastAPI HTTP server (entrypoint: `server.py`) handles webhook verification and message POSTs.
- `MessageProcessor` orchestrates extraction, validation, contact lookup, analysis and response generation.
- `AdvancedAIHandler` manages AI models (Gemini via google.generativeai when configured) with rotation and fallback to OpenRouter via `openrouter_client`.
- `EnhancedWhatsAppClient` encapsulates WhatsApp Graph API calls and enforces allowlist/production behavior.
- `PersonaManager` loads persona files (JSON) and allows runtime persona switching.
- MongoDB stores `conversations`, `contacts`, `messages`, `analytics` etc.

## Quickstart (local)

Checklist before starting:
- Python 3.10+ virtual environment
- MongoDB accessible at `MONGO_URL` or local MongoDB running
- WhatsApp Business API credentials (phone number id + token) if testing live sends
- Gemini / OpenRouter API keys as needed

1. Create and activate virtual environment (Windows PowerShell example):

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r agent/whats-app-agent/backend/requirements.txt
```

2. Copy `.env` template (example file located in `agent/whats-app-agent/backend/.env`) and fill in required values.

3. Start the app (development):

```powershell
cd agent/whats-app-agent/backend
& .venv\Scripts\python.exe server.py
# or with uvicorn if you want reload/host options:
& .venv\Scripts\python.exe -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

4. Expose your local server to WhatsApp using a tunnel (ngrok) and register webhook URL with the Facebook/Meta app.

## Environment variables (.env)

The backend reads many configuration values from environment variables. Important ones:

- `MONGO_URL` - MongoDB connection string (default: local)
- `DATABASE_NAME` - DB name (default: `whatsapp_agent_v2`)

- WhatsApp API
  - `WHATSAPP_ACCESS_TOKEN` - token for WhatsApp Business API
  - `WHATSAPP_PHONE_NUMBER_ID` - WhatsApp phone number id
  - `WHATSAPP_ALLOWED_RECIPIENTS` - optional comma-separated list of E.164 recipient numbers allowed for sending; empty means allow all (recommended to set in prod)
  - `DISABLE_WHATSAPP_SENDS` - set to `true` to short-circuit sends in dev/test

- Gemini (Google) keys
  - `GEMINI_API_KEY`, `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, ... - rotation-supported keys

- OpenRouter (fallback)
  - `OPENROUTER_API_KEY_1`, `OPENROUTER_API_KEY_2`, ...
  - `OPENROUTER_MODEL_ID` - model id (example: `mistralai/mistral-small-3.2-24b-instruct:free`)
  - `ENABLE_OPENROUTER_FALLBACK` - `true`/`false`

- Persona / behavior
  - `USER_PERSONALITY` - default system prompt if no persona selected
  - `AGENT_PERSONA` - name of persona to select at startup (optional)
  - `PERSONAS_DIR` - directory where persona JSON files live (default: `backend/personas`)

- App & owner
  - `WEBHOOK_VERIFY_TOKEN` - token used for verification handshake with Meta
  - `OWNER_WHATSAPP_NUMBER` - owner number for critical alerts
  - `APP_SECRET`, `APP_ID` - Facebook app values as required

- Retry & safety
  - `WHATSAPP_SEND_MAX_RETRIES` - default send retries (3)
  - `WHATSAPP_SEND_BACKOFF_BASE` - base seconds for exponential backoff (0.5)

Make sure to keep secrets out of git. Use a secrets manager or environment injection in CI/CD.

## Persona management (change agent by prompt)

The app includes a `PersonaManager` that loads JSON persona files and allows runtime switching. A persona JSON file looks like:

```json
{
  "name": "customer_support",
  "description": "Acme Co. customer support persona",
  "system_prompt": "You are Acme Co support. Be polite, follow policy X..."
}
```

Where to place persona files:
- Default directory: `agent/whats-app-agent/backend/personas/` (or override with `PERSONAS_DIR`).

Runtime APIs (FastAPI endpoints):
- `GET /api/personas` — list available personas and current selection
- `POST /api/personas/select?name=<persona>` — select persona by name (runtime switch)
- `POST /api/personas` — add a new persona (name, description, system_prompt fields)

After switching a persona the system prompt used in subsequent responses will reflect the persona. This enables switching from a personal assistant persona to a customer support persona by changing only a prompt.

## How AI fallback & key rotation work

- `AdvancedAIHandler` initialises multiple Gemini model instances if multiple `GEMINI_API_KEY*` values are present. On API errors (rate limits), it rotates to the next key.
- If all Gemini keys are exhausted (or the handler toggles `ai_available` false), the system will automatically try an OpenRouter fallback (`ENABLE_OPENROUTER_FALLBACK=true`).
- OpenRouter keys are also rotated by `OpenRouterClient`.
- All responses are post-processed and sanitized before sending.

## Running (dev vs production)

Development tips:
- Use `DISABLE_WHATSAPP_SENDS=true` while developing to avoid sending real WhatsApp messages.
- Use the simulator: `simulate_gemini_exhaustion.py` to validate fallback behavior.

Production checklist:
- Set `DISABLE_WHATSAPP_SENDS=false` or unset.
- Configure `WHATSAPP_ALLOWED_RECIPIENTS` or ensure recipients are registered with your WhatsApp Business account.
- Ensure MongoDB and Redis (optional) are production-ready.
- Ensure environment variables with keys/secrets are injected securely (Vault/Secrets Manager).
- Run with a process manager (systemd, Docker, Kubernetes) and monitor logs, uptime and retry metrics.

Run with uvicorn (production example):

```powershell
& .venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

Or create a Dockerfile and deploy to your container platform; ensure secrets are provided as environment variables at runtime.

## Testing & simulation

- Unit tests: add tests under `backend/tests/` (project contains a few test skeletons). Use `pytest`.
- Simulation script: `simulate_gemini_exhaustion.py` to force AI exhaustion and exercise OpenRouter fallback and message sending.
- Manual verification: use `ngrok` to expose a local endpoint and register webhook URL with WhatsApp app. Verify the webhook handshake via the browser GET request.

Example commands to run tests and simulation:

```powershell
# Run tests
& .venv\Scripts\python.exe -m pytest agent/whats-app-agent/backend/tests -q

# Run simulator
& .venv\Scripts\python.exe agent/whats-app-agent\backend\simulate_gemini_exhaustion.py
```

## Observability & logs

- Logs are written to `whatsapp_agent.log` and to stdout.
- The app logs important events: AI key rotation, owner notifications, WhatsApp send successes and failures, webhook verification, and background task scheduling.
- Use a centralized logging system (ELK, Datadog, Papertrail) in production.

## Troubleshooting (common errors)

- `Recipient phone number not in allowed list` (WhatsApp error #131030):
  - Causes: WhatsApp Business account is in a restricted testing mode or recipient not added to allowed list.
  - Fix: Add recipient to the allowed recipients or move to production WhatsApp Business API configuration. Set `WHATSAPP_ALLOWED_RECIPIENTS` if you want the app to validate locally.

- `All API keys exhausted` / repeated 429 errors:
  - Causes: All Gemini API keys hit rate limits or quota.
  - Fixes: Add more API keys, switch to paid plan, enable OpenRouter fallback, or throttle traffic. The app will notify `OWNER_WHATSAPP_NUMBER` once per exhaustion event.

- MongoDB connectivity errors:
  - Ensure `MONGO_URL` is correct and reachable from the host. Check credentials, firewall, and network routing.

- WhatsApp send errors 4xx (invalid token, forbidden):
  - Check `WHATSAPP_ACCESS_TOKEN` and `WHATSAPP_PHONE_NUMBER_ID`.

If you need a specific log trace, check `whatsapp_agent.log` for the timestamped structured logs.

## Security & secrets

- Never commit `.env` files or credentials to git.
- Use Secrets Manager (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault) for production.
- Rotate API keys periodically and remove unused keys.
- Limit `WHATSAPP_ACCESS_TOKEN` scope and use app secrets (APP_SECRET, APP_ID) properly.
- Restrict server access (security groups, firewalls) and use HTTPS in production.

## Scaling & production recommendations

- Use multiple workers or run multiple replicas behind a load balancer.
- Use a managed MongoDB cluster and configure indexes (the app creates indexes on startup).
- Use Redis for caching and rate-limiting to share load across replicas.
- Monitor API key usage and configure alerts for 429 spikes.

## Contributing

- Fork and open PRs against `main`.
- Follow the code style and add tests for new behaviors.

## FAQ / Tips

- Q: How do I switch the agent from personal assistant to customer support?
  - A: Create a persona JSON with the desired `system_prompt` and set `AGENT_PERSONA=name` or call `POST /api/personas/select?name=name` at runtime.

- Q: How do I test without sending messages?
  - A: Set `DISABLE_WHATSAPP_SENDS=true` in your `.env`.

## Contact

If you want me to add CI, Dockerfile, Kubernetes manifests, or example persona files, tell me which one and I'll add it.

---

Requirements coverage: this README covers setup, environment variables, persona management, running, testing, deployment guidance, troubleshooting and security.
# 🤖 AI WhatsApp Agent with Google Gemini

A sophisticated AI-powered WhatsApp automation system that handles conversations like a busy entrepreneur, using Google Gemini AI instead of OpenAI.

## ✨ Features

### 🧠 **Intelligent AI Conversation Handler**
- **Google Gemini Integration**: Advanced AI responses using Google's Gemini Pro model
- **Personality Mimicking**: Responds like a busy billionaire entrepreneur
- **Smart Classification**: Automatically categorizes messages (HIGH/MEDIUM/LOW priority, SPAM)
- **Context Awareness**: Maintains conversation history for coherent responses

### 📱 **WhatsApp Business API Integration**
- **Real-time Message Processing**: Instant webhook handling
- **Automatic Responses**: AI replies within seconds
- **Contact Management**: Identifies known vs unknown contacts
- **Name Collection**: Politely asks unknown contacts for their identity

### 📊 **Advanced Analytics Dashboard**
- **Real-time Statistics**: Live conversation metrics
- **Priority Breakdown**: Visual priority distribution
- **Conversation History**: Complete message threads
- **Search & Filter**: Find specific conversations quickly

### 📝 **Google Sheets Logging**
- **Automatic Data Export**: All conversations logged to Google Sheets
- **Comprehensive Tracking**: Timestamps, priorities, summaries, contact info
- **Analytics Ready**: Data formatted for analysis
- **Backup System**: Local logging fallback

### 🔔 **Smart Notification System**
- **Conversation Summaries**: Automatic end-of-chat summaries
- **Priority Alerts**: Immediate notifications for high-priority messages
- **Daily Reports**: Overview of all interactions

## 🚀 Quick Start

### Prerequisites
- WhatsApp Business API access token
- Google Gemini API key
- Google Sheets API credentials (optional)
- Python 3.8+
- Node.js 16+

### Installation

1. **Clone and Setup**
```bash
git clone <your-repo>
cd whatsapp-ai-agent

# Backend dependencies
cd backend
pip install -r requirements.txt

# Frontend dependencies  
cd ../frontend
yarn install
```

2. **Environment Configuration**

Update `/app/backend/.env`:
```env
WHATSAPP_ACCESS_TOKEN=your_whatsapp_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
# Primary Gemini API key (required)
GEMINI_API_KEY=your_primary_gemini_api_key
# Optional additional Gemini API keys for rotation/failover. The app will try them in order when rate limits/errors occur.
# You may provide up to 10 additional keys. Empty values are ignored.
GEMINI_API_KEY_1=your_optional_key_1
GEMINI_API_KEY_2=your_optional_key_2
GEMINI_API_KEY_3=
GEMINI_API_KEY_4=
GEMINI_API_KEY_5=
GEMINI_API_KEY_6=
GEMINI_API_KEY_7=
GEMINI_API_KEY_8=
GEMINI_API_KEY_9=
GEMINI_API_KEY_10=
GOOGLE_SHEETS_API_KEY=your_sheets_api_key
WEBHOOK_VERIFY_TOKEN=your_webhook_verify_token
```

3. **Start Services**
```bash
# Start both frontend and backend
sudo supervisorctl restart all

# Check status
sudo supervisorctl status
```

4. **Configure Webhook**
```bash
# Run the webhook setup script
python setup_webhook.py
```

## 🔧 Configuration

### WhatsApp Business API Setup

1. **Meta for Developers Account**
   - Visit https://developers.facebook.com
   - Create a new app for WhatsApp Business
   - Get your Access Token and Phone Number ID

2. **Webhook Configuration**
   ```
   Webhook URL: https://your-domain.com/api/webhook
   Verify Token: your_webhook_verify_token_123
   ```

### Google Gemini Setup

1. **Get API Key**
   - Visit https://makersuite.google.com/app/apikey
   - Create a new API key
   - Add to your `.env` file

2. **Personality Customization**
   
   Edit the `USER_PERSONALITY` in `.env`:
   ```env
   USER_PERSONALITY=You are a busy billionaire entrepreneur. Respond professionally but efficiently. Be direct, valuable, and respectful of time.
   ```

### Google Sheets Integration

1. **Setup Spreadsheet**
   - The system automatically creates a spreadsheet named "WhatsApp AI Agent Logs"
   - Or specify your own spreadsheet in the configuration

2. **Data Structure**
   ```
   Columns: Timestamp | Phone_Number | Contact_Name | Priority | 
            Category | Summary | Message_Count | Duration
   ```

## 📱 Usage

### Dashboard Access
- **URL**: http://localhost:3000
- **Features**: 
  - Live conversation monitoring
  - Message history viewing
  - Priority filtering
  - Search functionality
  - Real-time statistics

### API Endpoints

#### Webhook Endpoints
```bash
# Verify webhook (GET)
GET /api/webhook?hub.mode=subscribe&hub.challenge=test&hub.verify_token=your_token

# Receive messages (POST)
POST /api/webhook
```

#### Dashboard API
```bash
# Get statistics
GET /api/dashboard/stats

# Get conversations
GET /api/conversations?priority=HIGH&limit=50

# Get conversation messages  
GET /api/conversations/{conversation_id}/messages
```

## 🧪 Testing

### Local Testing with ngrok

1. **Install ngrok**
```bash
# Download from https://ngrok.com/download
npm install -g ngrok
```

2. **Expose Local Server**
```bash
# Start your backend server
python backend/server.py

# In another terminal, expose port 8001
ngrok http 8001
```

3. **Configure Webhook**
```bash
# Use the ngrok HTTPS URL
python setup_webhook.py
```

4. **Send Test Message**
```bash
# The setup script includes a test message feature
python setup_webhook.py
# Choose option 4: Send test message
```

### Production Testing

1. **Deploy to Cloud**
   - Heroku: `git push heroku main`
   - DigitalOcean: Use their app platform
   - Vercel: For frontend deployment

2. **Update Webhook URL**
   ```bash
   python setup_webhook.py
   # Use your production domain
   ```

## 📊 Monitoring & Logs

### Application Logs
```bash
# Backend logs
tail -f /var/log/supervisor/backend.out.log
tail -f /var/log/supervisor/backend.err.log

# Frontend logs  
tail -f /var/log/supervisor/frontend.out.log
```

### Conversation Logs
- **Google Sheets**: Automatic export to spreadsheet
- **Local Backup**: `/tmp/whatsapp_agent_conversations.jsonl`
- **Database**: MongoDB for real-time queries

## 🔒 Security & Privacy

### API Security
- **Webhook Verification**: Validates incoming requests
- **Token Authentication**: Secure WhatsApp API access
- **CORS Protection**: Configured for your domains only

### Data Privacy
- **Local Processing**: All AI processing on your servers
- **Secure Storage**: Encrypted database connections
- **Audit Trail**: Complete conversation logging

## 🚨 Troubleshooting

### Common Issues

**Webhook Not Working**
```bash
# Check if server is running
curl http://localhost:8001/

# Verify webhook endpoint
curl "http://localhost:8001/api/webhook?hub.mode=subscribe&hub.challenge=test&hub.verify_token=your_token"
```

**Messages Not Responding**
```bash
# Check backend logs
tail -f /var/log/supervisor/backend.err.log

# Verify API keys are set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Primary Gemini Key:', (os.getenv('GEMINI_API_KEY') or '')[:10] + '...')"
```

**Dashboard Not Loading**
```bash
# Check frontend status
sudo supervisorctl status frontend

# Restart if needed
sudo supervisorctl restart frontend
```

### Support Commands
```bash
# Restart all services
sudo supervisorctl restart all

# Check service status
sudo supervisorctl status

# View real-time logs
sudo supervisorctl tail -f backend
```

## 🎯 Advanced Configuration

### Custom AI Personality

Edit the personality in your `.env` file or modify the `AIConversationHandler` class:

```python
personality_prompt = """
You are a [YOUR ROLE]. 
- Respond with [YOUR STYLE]
- Handle [YOUR BUSINESS TYPE] inquiries
- Be [YOUR PERSONALITY TRAITS]
"""
```

### Priority Classification Rules

Modify `classify_message_priority()` in `server.py`:

```python
# Custom classification logic
if "investment" in message.lower() and amount_mentioned:
    return "HIGH", "business"
elif "fan" in sender_info.get("type", ""):
    return "LOW", "fan"
```

### Google Sheets Customization

Modify `google_sheets_integration.py` to add custom columns or formatting.

## 📈 Analytics & Reporting

### Built-in Analytics
- **Response Rate**: Percentage of messages answered
- **Priority Distribution**: Breakdown by priority levels
- **Peak Hours**: Busiest conversation times
- **Contact Growth**: New contacts over time

### Custom Reports
Export data from Google Sheets for advanced analysis:
- **Excel/CSV**: Download from Google Sheets
- **API Access**: Query the MongoDB database directly
- **Dashboard**: Build custom charts in the React frontend

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`  
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- **Documentation**: This README and inline code comments
- **Issues**: GitHub Issues for bug reports
- **Community**: Discord/Slack for discussions

---

**Built with ❤️ for busy entrepreneurs who value their time**

*Powered by Google Gemini AI, WhatsApp Business API, and modern web technologies*