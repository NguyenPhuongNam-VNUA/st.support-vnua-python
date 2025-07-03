from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyDIAOZIEKlPn4l3EQHGFd-qHJynO4pX1eQ")

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

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(temperature=0.4),
        contents=[prompt],
    )
    return response.text.strip()
