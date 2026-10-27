from langchain_core.documents import Document

ads = [
    Document(
        page_content="BMW 320d, дизель, пробег 140000 км, цена 14000 евро",
        metadata={"model": "BMW 320d", "year": 2015, "fuel": "diesel"},
    ),
    Document(
        page_content="BMW 320i, бензин, пробег 90000 км, цена 12000 евро",
        metadata={"model": "BMW 320i", "year": 2016, "fuel": "petrol"},
    ),
    Document(
        page_content="BMW 320xi, полный привод, пробег 110000 км, цена 13000 евро",
        metadata={"model": "BMW 320xi", "year": 2017, "fuel": "petrol"},
    ),
]

for ad in ads:
    print(ad.page_content)