"""Build index.html from site/template.html + site/core.js + site/records.json.  python build.py"""
import pathlib
root = pathlib.Path(__file__).parent
t = (root / 'site/template.html').read_text(encoding='utf-8')
t = t.replace('<meta charset="utf-8">\n', '', 1)
t = t.replace('__CORE__', (root / 'site/core.js').read_text(encoding='utf-8'))
t = t.replace('__RECORDS__', (root / 'site/records.json').read_text(encoding='utf-8'))
head, rest = t.split('</style>', 1)
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="26 and 27 unit 4-cubes packed in a 4-cube of side 2.942809, with an in-browser search you can rerun.">\n'
        '<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 16 16%27%3E%3Crect x=%271%27 y=%271%27 width=%2714%27 height=%2714%27 fill=%27none%27 stroke=%27%231f5fd1%27 stroke-width=%271.5%27/%3E%3Crect x=%275%27 y=%275%27 width=%276%27 height=%276%27 fill=%27%231f5fd1%27 transform=%27rotate(45 8 8)%27/%3E%3C/svg%3E">\n'
        + head + 'body { margin: 0; } [hidden] { display: none !important; }\n</style>\n</head>\n<body>\n' + rest + '\n</body>\n</html>\n')
(root / 'index.html').write_text(page, encoding='utf-8')
print('wrote index.html', len(page), 'bytes')
