import subprocess, os, re, html

with open('documentation.md') as f:
    md_content = f.read()

lines = md_content.split('\n')
html_lines = ['<!DOCTYPE html><html><head><meta charset="utf-8"><style>',
'body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #222; margin: 15mm; }',
'h1 { font-size: 16pt; border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 0; color: #111; }',
'h2 { font-size: 12pt; border-bottom: 1px solid #ccc; padding-bottom: 2px; margin-top: 12px; margin-bottom: 6px; color: #222; }',
'p { margin-top: 0; margin-bottom: 6px; text-align: justify; }',
'ul { margin-top: 0; margin-bottom: 6px; padding-left: 20px; }',
'li { margin-bottom: 3px; }',
'code { background-color: #eee; padding: 2px 4px; border-radius: 3px; font-family: monospace; font-size: 8.5pt; }',
'pre { background-color: #f4f4f4; padding: 6px; border-radius: 4px; font-family: monospace; font-size: 8.5pt; }',
'hr { border: 0; height: 1px; background: #ddd; margin: 10px 0; }',
'</style></head><body>']

in_pre = False
for line in lines:
    if line.startswith('```'):
        if in_pre:
            html_lines.append('</pre>')
            in_pre = False
        else:
            html_lines.append('<pre>')
            in_pre = True
    elif in_pre:
        html_lines.append(html.escape(line))
    elif line.startswith('# '):
        html_lines.append(f'<h1>{html.escape(line[2:])}</h1>')
    elif line.startswith('## '):
        html_lines.append(f'<h2>{html.escape(line[3:])}</h2>')
    elif line.startswith('---'):
        html_lines.append('<hr>')
    elif line.startswith('- '):
        content = html.escape(line[2:])
        content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'`(.*?)`', r'<code>\1</code>', content)
        html_lines.append(f'<li>{content}</li>')
    elif line.strip() == '':
        html_lines.append('')
    else:
        content = html.escape(line)
        content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'`(.*?)`', r'<code>\1</code>', content)
        html_lines.append(f'<p>{content}</p>')

html_lines.append('</body></html>')
with open('documentation.html', 'w') as f:
    f.write('\n'.join(html_lines))

cmd = ['libreoffice', '--headless', '--convert-to', 'pdf', 'documentation.html']
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout, res.stderr)
if os.path.exists('documentation.pdf'):
    print('SUCCESS: documentation.pdf created, size =', os.path.getsize('documentation.pdf'))
