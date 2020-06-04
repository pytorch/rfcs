"""An utility script that processes LaTeX comments in GitHub flavored
markdown file by appending the comment with LaTeX GitHub rendering
links.

The LaTeX comments are in the form

  <!--$LaTeX Formula$-->

where `LaTeX Formula` is any valid LaTeX string that does not contain
double minuses (`--`) and that
https://render.githubusercontent.com/render/math is able to render.

References:
  https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b
  https://github.github.com/gfm/#html-comment
"""
# Author: Pearu Peterson
# Created: May 2020

import os
import re
import sys

def symbolrepl(m):
    orig = m.string[m.start():m.end()]
    label = m.group('label')
    comment = '<!--:' + label + ':-->'
    label_map = dict(
        proposal=':large_blue_circle:',
        impl=':large_blue_diamond:',
    )
    return label_map.get(label, '') + comment
    if label == 'proposal':
        return ':large_blue_circle:' + comment
    return orig

def latexrepl(m):
    """
    Replace LaTeX formulas with LaTeX comments.
    """
    orig = m.string[m.start():m.end()]
    formula = m.group('formula').strip()
    dollars = m.string[m.start():m.start('formula')]
    is_latex_comment = m.string[:m.start()].endswith('<!--')
    if is_latex_comment:
        return orig
    return '<!--' + orig + '-->'

def formularepl(m):
    """
    Append LaTeX rendering images to LaTeX comments.
    """
    orig = m.string[m.start():m.end()]
    formula = m.group('formula').strip()
    dollars = m.string[m.start('dollars'):m.end('dollars')]

    print('formularepl:', formula, dollars)

    if not formula:
        return ''
    
    inline = len(dollars) == 1
    
    comment = '<!--' + dollars + formula + dollars + '-->'

    # Fix formula text for URI:
    formula = formula.replace('+', '%2B').replace('\n', '%0A')

    # img = '<img src="https://render.githubusercontent.com/render/math?math=' + formula + '" title="' + formula + '">'

    if 1:
        if inline:
            formula = '\\inline ' + formula
        img = '<img src="https://latex.codecogs.com/svg.latex?' + formula + '">'
    
    return comment + img


def main():
    filein = sys.argv[1]

    filename, ext = os.path.splitext(filein)
    assert ext == '.md'
    
    content = open(filein).read()

    count = 0
    for pattern, repl in [
            (r'[$]+(?P<formula>[^$]+)[$]+', latexrepl),
            (r'[<][!][-][-](?P<dollars>[$]+)(?P<formula>[^$]+)[$]+\s*[-][-][>](?P<prev>\s*[<]img\s+src[=]["].*?[>])?', formularepl),
            (r'(?P<prev>[:][^:]+[:]\s*)?[<][!][-][-][:](?P<label>.*?)[:][-][-][>]', symbolrepl),
    ]:
    
        content, _count = re.subn(
            pattern, repl,
            content,
            flags=re.S | re.M
        )
        count += _count

    if count > 0:
        print(f'Processed {count} items. Updating {filein}')
        f = open(filein, 'w')
        f.write(content)
        f.close()

if __name__ == '__main__':
    main()
