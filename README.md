# 🎓 VNUA Chatbot RAG API Microservice

Hệ thống Backend Microservice cho RAG Chatbot Học viện Nông nghiệp Việt Nam (VNUA), xây dựng chuẩn **Clean Architecture** với **FastAPI**, **PostgreSQL (`pgvector`)** và **Google Gemini API** (`gemini-2.5-flash` & `gemini-embedding-001`).

---

## 🏗️ Cấu trúc dự án chuẩn (Project Architecture)

```text
st.support-vnua-python/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── ask.py          # Chatbot RAG Route (/api/ask)
│   │       │   ├── questions.py    # Questions & Embeddings Route
│   │       │   └── tokens.py       # Token Count Route (/api/countToken)
│   │       └── router.py           # Tổng hợp Router API v1
│   ├── core/
│   │   ├── config.py               # Quản lý Cấu hình & Biến môi trường (Pydantic Settings)
│   │   └── database.py             # Kết nối PostgreSQL & pgvector
│   ├── models/
│   │   ├── question.py             # Model Question với Cột Vector
│   │   └── conversation.py         # Model Conversation (Lịch sử chat)
│   ├── schemas/
│   │   └── question.py             # Pydantic Request/Response Models
│   └── services/
│       ├── embedding_service.py    # Service tạo Gemini Embeddings & Count Token
│       └── rag_service.py          # Service RAG & Gemini 2.5 Flash LLM
│   └── main.py                     # Khởi tạo Ứng dụng FastAPI
├── main.py                         # File Entry-point chạy Uvicorn Server
├── .env                            # Biến môi trường local
├── .env.example                    # File mẫu biến môi trường
└── requirements.txt                # Danh sách Dependencies
```

---

## 🚀 Hướng dẫn khởi chạy

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Server
```bash
python main.py
```
Hoặc qua Uvicorn CLI:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 5001 --reload
```

### 3. Quản lý CSDL (Database Migrations với Alembic)

* **Áp dụng Migration mới nhất vào CSDL:**
  ```bash
  alembic upgrade head
  ```

* **Tự động tạo Migration mới khi thay đổi Model Python:**
  ```bash
  alembic revision --autogenerate -m "Mô tả thay đổi CSDL"
  ```

* **Rollback 1 phiên bản Migration:**
  ```bash
  alembic downgrade -1
  ```

### 4. Giao diện tài liệu API (Swagger UI)
Truy cập: **[http://localhost:5001/docs](http://localhost:5001/docs)**

