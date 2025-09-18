# WhatsApp AI Agent - Organized Directory Structure

This document describes the organized directory structure for the WhatsApp AI Agent project.

## Directory Structure

```
.
├── .git/                    # Git version control directory
├── .gitignore              # Git ignore file
├── .env.example            # Example environment configuration file
├── backend/                # Backend components
│   ├── check_and_register_webhook.py
│   ├── openrouter_client.py
│   └── utils.py
├── config/                 # Configuration files
│   └── api_keys.json       # API keys configuration (JSON format)
├── config_files/           # Configuration files (requirements, docker-compose, etc.)
│   ├── alembic.ini
│   ├── docker-compose.yml
│   └── requirements.txt
├── docs/                   # Documentation files
│   ├── *.md                # All markdown documentation files
│   └── *.patch             # Patch files
├── migrations/             # Database migration scripts
├── scripts/                # Utility scripts and startup scripts
│   ├── *.bat               # Batch files for Windows
│   ├── *.ps1               # PowerShell scripts
│   └── *.py                # Python utility scripts
├── server/                 # Main server application
│   ├── ai/                 # AI handling components
│   ├── clients/            # Client implementations
│   ├── db/                 # Database components
│   ├── models/             # Data models
│   ├── personas/           # Persona definitions
│   ├── repositories/       # Data repositories
│   ├── routes/             # API routes
│   ├── services/           # Core services
│   │   └── processor_modules/  # Message processor modules
│   ├── tools/              # Tools and utilities
│   ├── server.py           # Main server application
│   └── ...                 # Other server components
├── tests/                  # Test files
│   ├── test_*.py           # Test scripts
│   ├── verify_*.py         # Verification scripts
│   ├── check_*.py          # Check scripts
│   ├── debug_*.py          # Debug scripts
│   └── ...                 # Other test utilities
└── tools/                  # External tools (ngrok)
    ├── ngrok.exe           # Ngrok tunneling tool
    └── ...                 # Other tool files
```

## Directory Descriptions

### backend/
Contains core backend components including webhook handling and OpenRouter client.

### config/
Project configuration files including API keys in JSON format.

### config_files/
Additional configuration files including requirements, Docker configuration, and database migrations.

### docs/
All project documentation in markdown format, including implementation guides, summaries, and patch files.
- `autonomous_notification_system.md`: Documentation for the autonomous notification system

### migrations/
Database migration scripts and SQL initialization files.

### scripts/
Utility scripts for running the application, including batch files for Windows and PowerShell scripts.
- `initialize_notification_db.py`: Script to initialize the notification database schema

### server/
Main application server with modular components:
- `ai/`: AI handling and response generation
- `clients/`: API clients for external services
- `db/`: Database access and models
- `models/`: Data models and schemas
- `personas/`: JSON files defining different agent personas
- `repositories/`: Data access layer implementations
- `routes/`: API endpoints for webhooks and admin functions
- `services/`: Core business logic including message processing
- `services/processor_modules/`: Modular components for message processing
  - `autonomous_owner_notification_manager.py`: Autonomous notification system
  - `notification_db_schema.py`: Database schema definitions
  - `notification_monitoring.py`: Monitoring and alerting system
- `tools/`: Utility functions and classes

### tests/
All test files organized by function:
- Test scripts for various features
- Verification scripts for specific fixes
- Debug and check utilities
- `test_autonomous_notification_manager.py`: Unit tests for autonomous notification system
- `test_notification_monitoring.py`: Unit tests for notification monitoring
- `test_autonomous_notification_integration.py`: Integration tests for notification system

### tools/
External tools used by the project, primarily ngrok for tunneling.

## Configuration

The application can be configured in two ways:

1. **Environment Variables** (recommended for production):
   Copy `.env.example` to `.env` and fill in your values.

2. **JSON Configuration** (alternative method):
   Edit `config/api_keys.json` to include your API keys.

## Getting Started

1. Install dependencies: `pip install -r config_files/requirements.txt`
2. Configure environment variables or JSON configuration file
3. Initialize notification database: `python scripts/initialize_notification_db.py`
4. Run the server: `python server/server.py`
5. For development: `uvicorn server:app --reload --host 0.0.0.0 --port 8000`

## Running Tests

Execute tests with: `python -m pytest tests/ -v`

## Autonomous Notification System

The WhatsApp AI Agent now includes a fully autonomous notification system that makes all notification decisions without human intervention. Key features include:

- AI-driven decision making based on EOC confidence and importance scores
- Automatic notification sending via WhatsApp and email
- Persistent storage of all decisions for auditing and retraining
- Idempotency protection to prevent duplicate notifications
- Owner feedback collection for continuous improvement
- Comprehensive monitoring and alerting

For detailed information, see [Autonomous Notification System Documentation](docs/autonomous_notification_system.md).