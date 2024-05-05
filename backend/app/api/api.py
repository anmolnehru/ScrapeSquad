# app/api/api.py

from fastapi import APIRouter

from app.api.deps import fastapi_users, schemas, security
from app.api.routes import (
    data_orchestration,
    db_management,
    extractor,
    users,
)

api_router: APIRouter = APIRouter()

api_router.include_router(
    fastapi_users.get_auth_router(security.AUTH_BACKEND),
    prefix="/auth/jwt",
    tags=["auth"],
)
api_router.include_router(
    fastapi_users.get_register_router(schemas.UserRead, schemas.UserCreate),
    prefix="/auth",
    tags=["auth"],
)
api_router.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)
api_router.include_router(
    fastapi_users.get_verify_router(schemas.UserRead),
    prefix="/auth",
    tags=["auth"],
)
api_router.include_router(
    db_management.router, prefix="/db-management", tags=["db-management"]
)
api_router.include_router(
    users.router,
    prefix="/users",
    tags=["users"],
)
api_router.include_router(
    data_orchestration.router,
    prefix="/data_orchestration",
    tags=["data_orchestration"],
)
api_router.include_router(
    extractor.router,
    prefix="/extractor",
    tags=["extractor"],
)
