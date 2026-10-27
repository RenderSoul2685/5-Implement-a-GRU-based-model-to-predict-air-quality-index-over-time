import re

content_path = r"C:\Users\Kishan GK\.gemini\antigravity-ide\brain\df127ce2-9f36-4e79-b309-334e4de72d2e\.system_generated\steps\1086\content.md"
with open(content_path, "r", encoding="utf-8") as f:
    text = f.read()

# find script tags or fetch / download patterns
scripts = re.findall(r'<script.*?</script>', text, re.DOTALL)
print(f"Found {len(scripts)} script tags")
for i, s in enumerate(scripts):
    if any(k in s for k in ['site_', 'csv', 'parquet', 'data.vayuayan', 'download', 'fetch']):
        print(f"--- Script {i} ---")
        lines = s.split('\n')
        for l in lines:
            if any(k in l for k in ['fetch', 'url', 'csv', 'parquet', 'download', 'site_', 'data.']):
                print("  ", l[:120])
