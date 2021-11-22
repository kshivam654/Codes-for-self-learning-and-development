from tika import parser

raw = parser.from_file("a.pdf")
raw = str(raw['content'])

safe_text = raw.encode('utf-8', errors='ignore')
safe_text = str(safe_text)
safe_text = safe_text.replace(r'\n', '')
safe_text = safe_text.split('docx')[1]
print( safe_text )