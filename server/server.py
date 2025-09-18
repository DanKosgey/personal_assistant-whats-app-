from fastapi import FastAPI, Request, Header, HTTPException as FastAPIHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
try:
    # Use ORJSONResponse only if the optional orjson dependency is actually available
    import orjson as _orjson  # type: ignore
    from fastapi.responses import ORJSONResponse as DefaultResponseClass
except Exception:
    DefaultResponseClass = JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
import logging
import time
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

import os
import hmac
import hashlib
import re

from .utils import setup_logging
from pathlib import Path

from .config import config
from .database import db_manager
from .cache import cache_manager
from .ai import AdvancedAIHandler
from .clients import EnhancedWhatsAppClient
from .background import register_background_tasks
from .routes import router as routes_router

# Configure Sentry in production (lazy import to reduce cold start overhead)
if config.ENV == "production" and config.SENTRY_DSN:
    try:
        import sentry_sdk  # type: ignore
        from sentry_sdk.integrations.fastapi import FastApiIntegration  # type: ignore
        sentry_sdk.init(
            dsn=config.SENTRY_DSN,
            environment=config.ENV,
            integrations=[FastApiIntegration()],
            traces_sample_rate=0.2,
        )
    except Exception:
        pass

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.cache = cache_manager

    async def dispatch(self, request: Request, call_next: Callable):
        if config.RATE_LIMIT > 0:
            client_ip = request.client.host if request.client else "unknown"
            key = f"rate_limit:{client_ip}:{int(time.time() // 60)}"

            try:
                count = await self.cache.increment(key)
            except Exception:
                # If cache fails, allow the request to proceed (fail-open)
                return await call_next(request)

            if count > config.RATE_LIMIT:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"}
                )

            await self.cache.expire(key, 60)  # Reset after 1 minute

        return await call_next(request)


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
        return response

def create_app() -> FastAPI:
    setup_logging()
    
    app = FastAPI(
        title=config.APP_NAME,
        default_response_class=DefaultResponseClass,
        docs_url="/api/docs" if config.DEBUG else None,
        redoc_url="/api/redoc" if config.DEBUG else None,
        openapi_url="/api/openapi.json" if config.DEBUG else None,
    )

    # Security Middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=config.ALLOWED_HOSTS
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if config.RATE_LIMIT > 0:
        app.add_middleware(RateLimitMiddleware)

    # Compression Middleware (GZip always on; Brotli if available)
    app.add_middleware(GZipMiddleware, minimum_size=500)
    try:
        from brotli_asgi import BrotliMiddleware  # type: ignore
        app.add_middleware(BrotliMiddleware, quality=5)
    except Exception:
        # Brotli is optional; proceed if not installed/available
        pass

    # Lightweight request timing header
    app.add_middleware(TimingMiddleware)

    # Optional: Prometheus metrics exposure
    try:
        if config.ENABLE_METRICS:
            from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore
            Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except Exception:
        pass

    # Error Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return DefaultResponseClass(
            status_code=422,
            content={"detail": str(exc)},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return DefaultResponseClass(
            status_code=exc.status_code,
            content={"detail": str(exc.detail)},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        import traceback
        # Log full traceback and some request context for debugging
        tb = traceback.format_exc()
        try:
            body = await request.body()
            preview = body.decode('utf-8', errors='replace')[:200]
            # Simple redaction of phone numbers and emails
            import re as _re
            preview = _re.sub(r"\+?\d[\d\s\-()]{6,}\d", "<redacted:phone>", preview)
            preview = _re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<redacted:email>", preview)
            body_preview = preview
        except Exception:
            body_preview = '<unable to read body>'

        logger.error(
            "Unhandled exception for request %s %s - exc=%s\ntraceback=%s\nbody_preview=%s",
            request.method,
            request.url.path,
            str(exc),
            tb,
            body_preview,
        )
        logger.debug("Full exception traceback:\n%s", tb)
        return DefaultResponseClass(
            status_code=500,
            content={"detail": "Internal server error"}
        )

    # Health Check
    @app.get("/health")
    async def health_check():
        # Check if database is connected
        db_status = "unknown"
        db_connected = False
        try:
            if hasattr(db_manager, '_connected') and db_manager._connected:
                db_status = "connected"
                db_connected = True
            else:
                db_status = "disconnected"
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        return {
            "status": "healthy" if db_status == "connected" or db_status == "unknown" else "unhealthy",
            "version": "2.0.0",
            "env": config.ENV,
            "database": {
                "status": db_status,
                "connected": db_connected
            }
        }

    # --- WhatsApp/Meta webhook: GET verification and POST signature verification ---
    VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN", getattr(config, "WEBHOOK_VERIFY_TOKEN", "whatsapp_webhook_2025"))
    APP_SECRET = os.getenv("APP_SECRET", getattr(config, "APP_SECRET", ""))
    
    # For testing purposes, we can disable signature verification
    DISABLE_WEBHOOK_SIGNATURE_VERIFICATION = os.getenv("DISABLE_WEBHOOK_SIGNATURE_VERIFICATION", "false").lower() == "true"
    
    # internal flag to avoid spamming logs if APP_SECRET missing
    _app_secret_missing_logged = False

    def _verify_signature(body_bytes: bytes, signature_header: Optional[str]) -> bool:
        """
        Verify X-Hub-Signature-256 header (sha256=<hex>) using APP_SECRET.
        Returns True when signature matches; False otherwise.
        If APP_SECRET is not set, signature verification is considered disabled (development).
        """
        nonlocal _app_secret_missing_logged  # type: ignore
        
        # For testing purposes, we can disable signature verification
        if os.getenv("DISABLE_WEBHOOK_SIGNATURE_VERIFICATION", "false").lower() == "true":
            return True
            
        if not APP_SECRET:
            if not _app_secret_missing_logged:
                logger.warning("APP_SECRET not configured — webhook POST signature verification is disabled. This is unsafe in production.")
                _app_secret_missing_logged = True
            # In development, allow without signature. Change behavior if you want to enforce.
            return True

        if not signature_header:
            logger.debug("Missing X-Hub-Signature-26 header")
            return False
        prefix = "sha256="
        if not signature_header.startswith(prefix):
            logger.debug("Malformed X-Hub-Signature-256 header: %s", signature_header)
            return False
        sig = signature_header[len(prefix):]
        computed = hmac.new(APP_SECRET.encode(), body_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, computed)

    @app.get("/api/webhook", response_class=PlainTextResponse)
    async def verify_webhook(request: Request):
        """
        Respond to Meta's GET verification during subscription.
        Meta calls the callback URL with:
         - hub.mode=subscribe
         - hub.verify_token=<token>
         - hub.challenge=<challenge>
        Return the challenge (plain text) when token matches.
        """
        params = request.query_params
        mode = params.get("hub.mode")
        token = params.get("hub.verify_token") or params.get("hub.verify-token")
        challenge = params.get("hub.challenge")

        logger.info("Webhook verification request: mode=%s token_provided=%s challenge_present=%s", mode, bool(token), bool(challenge))

        if mode == "subscribe" and token and token == VERIFY_TOKEN and challenge:
            # Return the challenge exactly as plain text (HTTP 200)
            return PlainTextResponse(content=challenge, status_code=200)
        logger.warning("Webhook verification failed: mode=%s token_ok=%s", mode, token == VERIFY_TOKEN)
        raise FastAPIHTTPException(status_code=400, detail="Verification token mismatch or bad request")

    # Include routes AFTER webhook handlers to ensure proper routing
    app.include_router(routes_router, prefix="/api")

    @app.on_event("startup")
    async def startup():
        logger.info("🚀 Starting application...")
        try:
            # Shared HTTP client for connection pooling
            import httpx
            app.state.http_client = httpx.AsyncClient(timeout=30.0)
            logger.info("✅ HTTP client initialized")
            
            # Initialize database (optional)
            if not config.DISABLE_DB:
                logger.info("🔌 Attempting to connect to database...")
                try:
                    await db_manager.connect()
                    logger.info("✅ Database connected")
                except Exception as e:
                    logger.error(f"❌ Database connection failed: {e}")
                    raise
            else:
                logger.info("⏭️ Database disabled by DISABLE_DB flag")
            
            # Initialize cache
            try:
                await cache_manager.initialize()
                logger.info("✅ Cache initialized")
            except Exception as e:
                logger.warning(f"⚠️ Cache initialization failed: {e}")
            
            # Initialize AI handler
            try:
                app.state.ai = AdvancedAIHandler(config=config, http_client=app.state.http_client)
                logger.info("✅ AI handler initialized")
            except Exception as e:
                logger.warning(f"⚠️ AI handler initialization failed: {e}")
            
            # Initialize WhatsApp client
            try:
                app.state.whatsapp = EnhancedWhatsAppClient(http_client=app.state.http_client)
                logger.info("✅ WhatsApp client initialized")
            except Exception as e:
                logger.warning(f"⚠️ WhatsApp client initialization failed: {e}")
            
            # Initialize Message Processor with Memory System
            try:
                from .services.processor_refactored import MessageProcessor
                app.state.message_processor = MessageProcessor(ai=app.state.ai, whatsapp=app.state.whatsapp)
                logger.info("✅ Message processor with memory system initialized")
            except Exception as e:
                logger.warning(f"⚠️ Message processor initialization failed: {e}")
            
            # Initialize Memory API Routes
            try:
                from .routes.memory import set_message_processor
                set_message_processor(app.state.message_processor)
                logger.info("✅ Memory API routes initialized")
            except Exception as e:
                logger.warning(f"⚠️ Memory API routes initialization failed: {e}")
            
            # Register background tasks
            try:
                register_background_tasks(app)
                logger.info("✅ Background tasks registered")
            except Exception as e:
                logger.warning(f"⚠️ Background tasks registration failed: {e}")
            
            logger.info(f"🚀 Application started successfully in {config.ENV} mode")
        except Exception as e:
            logger.error(f"❌ Failed to start application: {str(e)}")
            raise

    @app.on_event("shutdown")
    async def shutdown():
        try:
            if not config.DISABLE_DB:
                await db_manager.close()
                
                # Close profile database
                try:
                    from .db.profile_db import profile_db
                    await profile_db.close()
                except Exception:
                    pass
                    
            await cache_manager.close()
            try:
                client = getattr(app.state, "http_client", None)
                if client is not None:
                    await client.aclose()
            except Exception:
                pass
            logger.info("✅ Application shutdown complete")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {str(e)}")

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # Prefer uvloop when available for better performance
    _loop = None
    try:
        import uvloop  # type: ignore
        uvloop.install()
        _loop = "uvloop"
    except Exception:
        _loop = None
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.PORT,
        workers=4 if config.ENV == "production" else 1,
        proxy_headers=True,
        forwarded_allow_ips="*",
        access_log=True,
        loop=_loop or "auto",
    )
