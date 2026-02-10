from chunker import chunk_text

with open("./data/cat-facts.txt", "r", encoding="utf-8") as f:
    text = f.read()


print(chunk_text(text=text, max_length=1))