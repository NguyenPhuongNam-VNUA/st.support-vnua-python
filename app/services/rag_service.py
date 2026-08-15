from google import genai
from google.genai import types
from app.core.config import settings
from app.services.embedding_service import CustomGeminiEmbeddings
from typing import List, Dict, Tuple, Optional

client = genai.Client(api_key=settings.GOOGLE_API_KEY)

def generate_rag_answer(question: str, context: str, history: List[Dict]) -> str:
    """Gọi Gemini 2.5 Flash sinh câu trả lời RAG dựa trên Context và Lịch sử trò chuyện."""
    prompt = f"""
    NỘI DUNG THAM KHẢO: {context}

    CÂU HỎI CỦA SINH VIÊN: {question}
    """

    chat_history = []
    for msg in history:
        if msg.get('role') == 'user':
            chat_history.append(types.Content(role='user', parts=[types.Part(text=msg.get('text', ''))]))
        elif msg.get('role') == 'assistant':
            chat_history.append(types.Content(role='model', parts=[types.Part(text=msg.get('text', ''))]))

    try:
        chat = client.chats.create(
            model="gemini-2.5-flash",
            history=chat_history,
            config=types.GenerateContentConfig(
                system_instruction="""
                Bạn là trợ lý ảo chính thức giúp sinh viên Học viện Nông nghiệp Việt Nam (VNUA) trả lời thắc mắc. 
                Nhiệm vụ của bạn là trả lời các câu hỏi của sinh viên dựa trên thông tin được cung cấp một cách chính xác, ngắn gọn và rõ ràng. 
                Chủ đề trả lời của bạn là những vấn đề liên quan sinh viên, học vụ, chính sách, giấy tờ, nhà trường, thầy cô…
                - Trả lời "CÂU HỎI" chỉ dựa trên "NỘI DUNG THAM KHẢO" đã cho, tuyệt đối không tự bịa đặt thông tin không có trong ngữ cảnh.
                - Nếu "CÂU HỎI" nằm ngoài chủ đề sinh viên/nhà trường hãy trả lời: "Xin lỗi, hệ thống hiện chưa hỗ trợ chủ đề này."
                - Nếu "CÂU HỎI" liên quan đến chủ đề nhưng là câu hỏi mơ hồ, chưa rõ ràng hãy gợi ý lại câu hỏi rõ ràng hơn (khoảng 1-2 gợi ý).
                - Nếu "CÂU HỎI" liên quan đến chủ đề mà chưa có dữ liệu trong hệ thống, hãy trả lời chính xác cụm từ:
                  "Câu hỏi này hiện chưa có thông tin trong hệ thống. Cảm ơn bạn, mình sẽ cập nhật sớm!"
                """,
                temperature=0.6,
            ),
        )
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[✗] Lỗi khi gọi Gemini: {e}")
        return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."

def build_context(results: List[Tuple[Dict, float]]) -> Dict:
    """Xây dựng chuỗi Context từ kết quả Retrieval."""
    context_parts = []
    score = 0.0
    id_val = ""
    content = ""

    for i, (item, score_val) in enumerate(results, 1):
        topic = item.get("topic", "Chưa rõ")
        answer = item.get("answer", "Chưa có câu trả lời")
        question = item.get("question", "").strip()

        id_val = item.get("id", "")
        score = score_val
        content = question

        context_parts.append(
            f"[Thông tin tham khảo #{i}]:\n"
            f"- Chủ đề: {topic}\n"
            f"- Hỏi: {question}\n"
            f"- Trả lời: {answer}\n"
        )

    return {
        "context": "\n".join(context_parts).strip(),
        "score": 1 - score if score else 0.0,
        "id": id_val,
        "content": content
    }
