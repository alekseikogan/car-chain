from langchain_text_splitters import RecursiveCharacterTextSplitter

text = (
    "BMW 320d 2015 года, дизель, пробег 140000 км, цена 14000 евро. "
    "Мотор 2.0 хорошо тянет по трассе и расходует около 5 литров. "
    "Коробка автомат переключается плавно, подвеска мягкая. "
    "Салон без прокуров, сервисная история есть, последний раз меняли масло и колодки. "
    "Минус: зимой долго греется, а турбина на таком пробеге уже требует проверки."
)

splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
# chunk_size — верхняя граница длины куска в символах.
# chunk_overlap — сколько символов с конца куска повторяется в начале следующего. 
# Фраза, разрезанная границей, целиком остаётся хотя бы в одном куске.

chunks = splitter.split_text(text)

print("chunks:", len(chunks))
for chunk in chunks:
    print("---")
    print(chunk)