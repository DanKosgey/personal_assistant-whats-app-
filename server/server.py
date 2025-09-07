from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
try:
    from fastapi.responses import ORJSONResponse as DefaultResponseClass
except Exception:
    DefaultResponseClass = JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
import logging
import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

from .utils import setup_logging
from pathlib import Path

from .config import config
from .db import db_manager
from .cache import cache_manager
from .ai import AdvancedAIHandler
from .clients import EnhancedWhatsAppClient
from .background import register_background_tasks
from .routes import router as routes_router

# Configure Sentry in production
if config.ENV == "production" and config.SENTRY_DSN:
    sentry_sdk.init(
        dsn=config.SENTRY_DSN,
        environment=config.ENV,
        integrations=[FastApiIntegration()],
        traces_sample_rate=0.2,
    )

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
        return {
            "status": "healthy",
            "version": "2.0.0",
            "env": config.ENV
        }

    app.include_router(routes_router, prefix="/api")

    @app.on_event("startup")
    async def startup():
        try:
            # Shared HTTP client for connection pooling
            import httpx
            app.state.http_client = httpx.AsyncClient(timeout=30.0)
            # Initialize database (optional)
            if not config.DISABLE_DB:
                await db_manager.connect()
                logger.info("✅ Database connected")
            else:
                logger.info("⏭️ Database disabled by DISABLE_DB flag")
            
            # Initialize cache
            await cache_manager.initialize()
            logger.info("✅ Cache initialized")
            
            # Initialize AI handler
            app.state.ai = AdvancedAIHandler(config=config, http_client=app.state.http_client)
            logger.info("✅ AI handler initialized")
            
            # Initialize WhatsApp client
            app.state.whatsapp = EnhancedWhatsAppClient(http_client=app.state.http_client)
            logger.info("✅ WhatsApp client initialized")
            
            # Register background tasks
            register_background_tasks(app)
            logger.info("✅ Background tasks registered")
            
            logger.info(f"🚀 Application started in {config.ENV} mode")
        except Exception as e:
            logger.error(f"❌ Failed to start application: {str(e)}")
            raise

    @app.on_event("shutdown")
    async def shutdown():
        try:
            if not config.DISABLE_DB:
                await db_manager.close()
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
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.PORT,
        workers=4 if config.ENV == "production" else 1,
        proxy_headers=True,
        forwarded_allow_ips="*",
        access_log=True,
    )
