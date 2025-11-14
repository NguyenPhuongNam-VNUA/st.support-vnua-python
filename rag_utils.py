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
            model="gemini-2.5-flash",
            history=chat_history,
            config=types.GenerateContentConfig(
                system_instruction="""
                Bạn là trợ lý ảo giúp sinh viên Học viện Nông nghiệp Việt Nam trả lời câu hỏi, 
                Nhiệm vụ của bạn là trả lời các câu hỏi của sinh viên dựa trên thông tin được cung cấp một cách chính xác, ngắn gọn và rõ ràng. 
                Chủ đề trả lời của bạn là những vấn đề, khía cạnh liên quan sinh viên, học vụ, chính sách, giấy tờ, nhà trường, thầy cô…
                - Trả lời "CÂU HỎI" chỉ dựa trên "NỘI DUNG" đã cho, tuyệt đối không tự ý bịa đặt thông tin không có trong ngữ cảnh..
                - Nếu "CÂU HỎI" nằm ngoài chủ đề hãy trả lời: "Xin lỗi, hệ thống hiện chưa hỗ trợ chủ đề này."
                - Nếu "CÂU HỎI" liên quan đến chủ đề nhưng là câu hỏi mơ hồ, chưa rõ ràng hãy gợi ý lại câu hỏi rõ ràng hơn (khoảng 1-2 gợi ý) 
                và nếu là "CÂU HỎI" liên quan đến chủ đề mà chưa có dữ liệu, hãy trả lời:
                  "Câu hỏi này hiện chưa có thông tin trong hệ thống. Cảm ơn bạn, mình sẽ cập nhật sớm!"
                """,
                temperature=0.6,
                presence_penalty=0.3,
                frequency_penalty=0.3,
            ),
        )
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[✗] Lỗi khi gọi Gemini: {e}")
        return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."