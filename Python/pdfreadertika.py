from tika import parser

raw = parser.from_file("example.pdf")
raw = str(raw['content'])

safe_text = raw.encode('utf-8', errors='ignore')
safe_text = str(safe_text).replace("\n", "").replace("\\", "")
print( safe_text )