with open(r"C:\Users\Kishan GK\.gemini\antigravity-ide\brain\df127ce2-9f36-4e79-b309-334e4de72d2e\.system_generated\steps\1086\content.md", "r", encoding="utf-8") as f:
    text = f.read()

idx = 0
while True:
    pos = text.find("const fileName", idx)
    if pos == -1:
        break
    print("=== Found at pos", pos, "===")
    start = max(0, pos - 400)
    end = min(len(text), pos + 800)
    print(text[start:end])
    idx = pos + 1
