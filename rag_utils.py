import os
from dotenv import load_dotenv
import concurrent.futures
from google import generativeai as genai
from google.generativeai import GenerativeModel

# Load biến môi trường và cấu hình API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = GenerativeModel("gemini-2.0-flash")

def call_gemini(prompt):
    return model.generate_content(prompt).text.strip()

def generate_rag_answer(question, context):
    prompt = f"""
    Bạn là trợ lý ảo giúp sinh viên Học viện Nông nghiệp Việt Nam trả lời câu hỏi.

    - Trả lời câu hỏi bên dưới chỉ dựa trên "NỘI DUNG" đã cho phía dưới.
    - Nếu câu hỏi nằm ngoài chủ đề sinh viên, học vụ, chính sách, giấy tờ, nhà trường, thầy cô… hoặc không thể trả lời từ nội dung, hãy trả lời:
      "Xin lỗi, tôi hiện chưa hỗ trợ chủ đề này."
    - Nếu câu hỏi liên quan nhưng là câu hỏi mơ hồ, chưa rõ ràng hãy gợi ý lại câu hỏi rõ ràng hơn (những câu tương đồng 25%-40%) (khoảng 1-2 gợi ý) 
    và nếu là câu hỏi liên quan mà chưa có dữ liệu (độ tương đồng rất thấp), hãy trả lời:
      "Câu hỏi này hiện chưa có thông tin trong hệ thống. Cảm ơn bạn, mình sẽ cập nhật sớm!"

    NỘI DUNG: {context}

    CÂU HỎI: {question}
    """

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_gemini, prompt)
            return future.result(timeout=10)
    except concurrent.futures.TimeoutError:
        return "Xin lỗi, hệ thống đang phản hồi chậm. Vui lòng thử lại sau."
    except Exception as e:
        print(f"[✗] Lỗi khi gọi Gemini: {e}")
        return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
