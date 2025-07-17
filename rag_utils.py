from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

# Cấu hình API key
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_rag_answer(question, context, history):
    prompt = f"""
    NỘI DUNG: {context}

    CÂU HỎI: {question}
    """

    chat_history = []
    for msg in history:
        if msg['role'] == 'user':
            chat_history.append(types.Content(role='user', parts=[types.Part(text=msg['text'])]))
        elif msg['role'] == 'assistant':
            chat_history.append(types.Content(role='model', parts=[types.Part(text=msg['text'])]))

    try:
        chat = client.chats.create(
            model="gemini-2.0-flash",
            history=chat_history,
            config=types.GenerateContentConfig(
                system_instruction="""
                Bạn là trợ lý ảo giúp sinh viên Học viện Nông nghiệp Việt Nam trả lời câu hỏi, 
                xưng hô "bạn" với người dùng và trả lời câu hỏi một cách tự nhiên, dễ hiểu.
                Chủ đề trả lời của bạn là những vấn đề, khía cạnh liên quan sinh viên, học vụ, chính sách, giấy tờ, nhà trường, thầy cô…
                - Trả lời câu hỏi bên dưới chỉ dựa trên "NỘI DUNG" đã cho.
                - Nếu câu hỏi nằm ngoài chủ đề hãy trả lời: "Xin lỗi, tôi hiện chưa hỗ trợ chủ đề này."
                - Nếu câu hỏi liên quan đến chủ đề nhưng là câu hỏi mơ hồ, chưa rõ ràng hãy gợi ý lại câu hỏi rõ ràng hơn (khoảng 1-2 gợi ý) 
                và nếu là câu hỏi liên quan đến chủ đề mà chưa có dữ liệu, hãy trả lời:
                  "Câu hỏi này hiện chưa có thông tin trong hệ thống. Cảm ơn bạn, mình sẽ cập nhật sớm!"
                """,
                temperature=0.5,
                presence_penalty=0.3,
                frequency_penalty=0.3,
            ),
        )
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[✗] Lỗi khi gọi Gemini: {e}")
        return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
