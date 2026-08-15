from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlmodel import Session, select

from app.core.database import get_session
from app.core.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)
from app.models.user import User
from app.schemas.auth import LoginRequest, Token, UserCreate, UserResponse

router = APIRouter()

# OAuth2 Scheme cho FastAPI Swagger UI (/docs)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    session: Annotated[Session, Depends(get_session)],
) -> User:
    """Dependency trích xuất và xác thực PyJWT Token từ Header Authorization: Bearer <token>"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token không hợp lệ hoặc đã hết hạn",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    payload = decode_access_token(token)
    if not payload:
        raise credentials_exception

    username: str = payload.get("sub")
    if not username:
        raise credentials_exception

    statement = select(User).where(User.username == username)
    user = session.exec(statement).first()

    if not user:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tài khoản đã bị vô hiệu hóa",
        )

    return user


@router.post("/login", response_model=Token, summary="Đăng nhập lấy PyJWT Access Token (Form Data / Swagger UI)")
def login_form(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Annotated[Session, Depends(get_session)],
):
    """
    API Đăng nhập dùng Form Data (Chuẩn OAuth2 - dùng trực tiếp cho nút Authorize trên Swagger UI /docs).
    """
    statement = select(User).where(User.username == form_data.username)
    user = session.exec(statement).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tài khoản hoặc mật khẩu không chính xác",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tài khoản đã bị vô hiệu hóa",
        )

    access_token = create_access_token(subject=user.username)
    return Token(access_token=access_token, token_type="bearer")


@router.post("/login/json", response_model=Token, summary="Đăng nhập lấy PyJWT Access Token (JSON Body)")
def login_json(
    json_data: LoginRequest,
    session: Annotated[Session, Depends(get_session)],
):
    """
    API Đăng nhập dùng JSON Body (dùng cho Frontend React / Mobile app).
    """
    statement = select(User).where(User.username == json_data.username)
    user = session.exec(statement).first()

    if not user or not verify_password(json_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tài khoản hoặc mật khẩu không chính xác",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tài khoản đã bị vô hiệu hóa",
        )

    access_token = create_access_token(subject=user.username)
    return Token(access_token=access_token, token_type="bearer")


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Đăng ký tài khoản người dùng mới",
)
def register(
    user_in: UserCreate,
    session: Annotated[Session, Depends(get_session)],
):
    """Đăng ký tài khoản mới với mật khẩu được hash bằng bcrypt."""
    # Kiểm tra trùng lặp username
    statement_username = select(User).where(User.username == user_in.username)
    if session.exec(statement_username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên đăng nhập đã được sử dụng",
        )

    # Kiểm tra trùng lặp email
    statement_email = select(User).where(User.email == user_in.email)
    if session.exec(statement_email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email đã được sử dụng",
        )

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name,
        is_active=True,
    )

    session.add(user)
    session.commit()
    session.refresh(user)

    return user


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Lấy thông tin người dùng hiện tại (Yêu cầu JWT Token)",
)
def get_me(current_user: Annotated[User, Depends(get_current_user)]):
    """Lấy thông tin tài khoản đang đăng nhập từ Header Authorization Bearer PyJWT."""
    return current_user
