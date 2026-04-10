import re

def clean_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    
    # Danh sách các chuỗi cần xóa (dạng exact hoặc contains)
    remove_texts = [
        "QUỐC HỘI",
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
        "Độc lập - Tự do - Hạnh phúc",
        "Luật số: /2025/QH15",
        "Hà Nội, ngày tháng năm 2025",
        "Dự thảo",
        "01/12/2025",
        "CHỦ TỊCH QUỐC HỘI",
        "Trần Thanh Mẫn",
        "The following table:",
        "Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam",
        "Quốc hội ban hành Luật Trí tuệ nhân tạo",
        "Luật này được Quốc hội nước Cộng hòa xã hội chủ nghĩa Việt Nam",
        "| --- | --- |",
        "|     |     |",
        "|     |",
        "|",
    ]

    for line in lines:
        line = line.strip()
        
        # Xóa thẻ HTML (span, br, v.v)
        line = re.sub(r'<[^>]+>', '', line)
        
        # Bỏ qua các dòng rỗng
        if not line:
            continue
            
        # Loại bỏ gạch chân của markdown (e.g. "_Căn cứ_")
        line = line.replace('_', '')
        
        # Kiểm tra xem dòng có chứa các từ khóa cần bỏ không
        skip = False
        for rm in remove_texts:
            if rm in line:
                skip = True
                break
        if skip:
            continue
            
        # Làm sạch các list item sai do markdown
        line = re.sub(r'^\- \- \- ', '', line)
        line = re.sub(r'^(\d+)\\?\.', r'\1.', line) # 1\. -> 1.
        line = re.sub(r'^\- (\d+)\.', r'\1.', line) # - 1. -> 1.
        
        # Chuẩn hóa khoảng trắng
        line = re.sub(r'\s+', ' ', line).strip()
        
        if line:
            cleaned_lines.append(line)

    # Nối lại các dòng
    merged_lines = []
    i = 0
    while i < len(cleaned_lines):
        curr = cleaned_lines[i]
        
        # Trong khi dòng hiện tại kết thúc bằng các ký tự cần nối và vẫn còn dòng tiếp theo
        while i < len(cleaned_lines) - 1 and (curr.endswith(';') or curr.endswith(',') or curr.endswith('và') or curr.endswith('hoặc') or curr.endswith('hay')):
            i += 1
            curr = curr + " " + cleaned_lines[i]
            
        merged_lines.append(curr)
        i += 1
        
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in merged_lines:
            f.write(line + "\n\n")
            
clean_data("data/du-thao-Luat-AI.md", "data/luat_ai_cleaned.md")
print("Cleaned data saved to data/luat_ai_cleaned.md")
