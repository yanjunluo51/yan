import re, os, subprocess, pymupdf
SRC='/home/user/yan/revised/CAIE_main_revised.tex'
AUX='/home/user/yan/revised/CAIE_main_revised.aux'
OUT='/tmp/claude-0/-home-user-yan/5364adbd-c49d-5b0b-88c9-e1c4fd24fd61/scratchpad/docx/pandoc_in.tex'
FIG='/tmp/claude-0/-home-user-yan/5364adbd-c49d-5b0b-88c9-e1c4fd24fd61/scratchpad/docx/fig'
os.makedirs(FIG,exist_ok=True)
s=open(SRC).read()

# ---- labels from aux
labels={}
for m in re.finditer(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}',open(AUX).read()):
    labels[m.group(1)]=m.group(2)

# ---- bibliography: parse bibitems
bib=re.search(r'\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}',s,re.S).group(1)
items=re.findall(r'\\bibitem\[\{(.*?)\}\]\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)',bib,re.S)
cite={}
for lab,key,text in items:
    m=re.match(r'(.*?)\((\d{4}[a-z]?)\)',lab)
    short=m.group(1).replace('~',' ').strip(); year=m.group(2)
    cite[key]=(short,year)
def citet(keys):
    keys=[k.strip() for k in keys.split(',')]
    groups=[]
    for k in keys:
        a,y=cite[k]
        if groups and groups[-1][0]==a: groups[-1][1].append(y)
        else: groups.append([a,[y]])
    return '; '.join(f"{a} ({', '.join(ys)})" for a,ys in groups)
def citep(keys):
    keys=[k.strip() for k in keys.split(',')]
    parts=[]
    for k in keys:
        a,y=cite[k]
        if parts and parts[-1][0]==a: parts[-1][1].append(y)
        else: parts.append([a,[y]])
    return '('+'; '.join(f"{a}, {', '.join(ys)}" for a,ys in parts)+')'
s=re.sub(r'\\citet\{([^}]*)\}',lambda m:citet(m.group(1)),s)
s=re.sub(r'\\citep\{([^}]*)\}',lambda m:citep(m.group(1)),s)

# ---- references list -> plain section
def bibblock(m):
    out=['\\section*{References}','']
    for lab,key,text in items:
        t=re.sub(r'\s+',' ',text).strip()
        out.append(t+'\n')
    return '\n'.join(out)
s=re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',bibblock,s,flags=re.S)
s=s.replace('\\expandafter\\ifx\\csname natexlab\\endcsname\\relax\\def\\natexlab#1{#1}\\fi','')

# ---- equations: number them
def eqnum(body):
    m=re.search(r'\\label\{([^}]*)\}',body)
    n=labels.get(m.group(1),'?') if m else None
    body=re.sub(r'\\label\{[^}]*\}','',body).strip()
    return body,n
def repl_eq(m):
    body,n=eqnum(m.group(1))
    return f"\n\\[ {body} \\qquad ({n}) \\]\n"
s=re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}',repl_eq,s,flags=re.S)
def repl_ml(m):
    body,n=eqnum(m.group(1))
    body=body.replace('\\\\','\\\\ &')
    return f"\n\\[ \\begin{{aligned}} & {body} \\end{{aligned}} \\qquad ({n}) \\]\n"
s=re.sub(r'\\begin\{multline\}(.*?)\\end\{multline\}',repl_ml,s,flags=re.S)

# ---- refs
s=re.sub(r'\\ref\{([^}]*)\}',lambda m:labels.get(m.group(1),'??'),s)
s=s.replace('~',' ')

# ---- captions with numbers
cnt={'figure':0,'table':0}
def fix_float(m):
    env=m.group(1); body=m.group(2)
    kind='figure' if env.startswith('figure') else 'table'
    cm=re.search(r'\\label\{([^}]*)\}',body)
    num=labels.get(cm.group(1)) if cm else None
    name='Figure' if kind=='figure' else 'Table'
    body=re.sub(r'\\caption\{',lambda mm:'\\caption{'+f'{name} {num}. ',body,count=1)
    body=re.sub(r'\\label\{[^}]*\}','',body)
    return f'\\begin{{{env}}}{body}\\end{{{env}}}'
s=re.sub(r'\\begin\{(figure\*?|table\*?)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}',fix_float,s,flags=re.S)

# ---- tikz figures -> standalone PNG
pre=open('/home/user/yan/revised/sections/00_preamble.tex').read()
macros='\n'.join(l for l in pre.splitlines() if l.startswith('\\newcommand'))
tikzs=re.findall(r'\\resizebox\{[^}]*\}\{!\}\{%\s*(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})\}',s,re.S)
for i,t in enumerate(tikzs):
    doc=('\\documentclass[border=4pt]{standalone}\n\\usepackage{amsmath,amssymb,tikz}\n\\usepackage{newtxtext,newtxmath}\n'
         '\\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,fit,backgrounds}\n'+macros+'\n\\begin{document}\n'+t+'\n\\end{document}\n')
    open(f'{FIG}/tikz{i}.tex','w').write(doc)
    r=subprocess.run(['pdflatex','-interaction=nonstopmode','-output-directory',FIG,f'{FIG}/tikz{i}.tex'],capture_output=True,text=True)
    if r.returncode!=0:
        doc=doc.replace('\\usepackage{newtxtext,newtxmath}\n','\\usepackage{mathptmx}\n')
        open(f'{FIG}/tikz{i}.tex','w').write(doc)
        subprocess.run(['pdflatex','-interaction=nonstopmode','-output-directory',FIG,f'{FIG}/tikz{i}.tex'],capture_output=True)
    d=pymupdf.open(f'{FIG}/tikz{i}.pdf'); d[0].get_pixmap(dpi=250).save(f'{FIG}/tikz{i}.png')
k=[0]
def tikzrepl(m):
    i=k[0]; k[0]+=1
    return f'\\includegraphics[width=6.3in]{{{FIG}/tikz{i}.png}}'
s=re.sub(r'\\resizebox\{[^}]*\}\{!\}\{%\s*\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}\}',tikzrepl,s,flags=re.S)
# ---- PDF figures -> PNG
def pdffig(m):
    opt,path=m.group(1),m.group(2)
    src='/home/user/yan/revised/'+path
    png=f'{FIG}/'+os.path.basename(path).replace('.pdf','.png')
    d=pymupdf.open(src); d[0].get_pixmap(dpi=250).save(png)
    w='6.3in' if 'textwidth' in (opt or '') else '3.4in'
    return f'\\includegraphics[width={w}]{{{png}}}'
s=re.sub(r'\\includegraphics(\[[^\]]*\])?\{(figures/[^}]*\.pdf)\}',pdffig,s)

# ---- algorithm -> numbered paragraphs
def algo(m):
    body=m.group(1)
    cap=re.search(r'\\caption\{(.*?)\}\n',body).group(1)
    lines=[l.strip() for l in body.splitlines()]
    out=[f'\\noindent\\textbf{{Algorithm 1.}} {cap}\n']
    n=0; depth=0
    def cm(t):
        return re.sub(r'\\Comment\{(.*)\}$',r' $\\triangleright$ \\textit{\1}',t)
    for l in lines:
        if l.startswith('\\Require'): out.append('\\noindent\\textbf{Input} '+cm(l[8:].strip())+'\n'); continue
        if l.startswith('\\Ensure'): out.append('\\noindent\\textbf{Output} '+cm(l[7:].strip())+'\n'); continue
        kw=None
        mm=re.match(r'\\(State|If|Else|EndIf|Return)\b(.*)',l)
        if not mm: continue
        typ,rest=mm.group(1),mm.group(2).strip()
        if typ in('Else','EndIf'): depth-=1
        n+=1
        ind='\\hspace*{'+str(1.2*depth)+'em}' if depth>0 else ''
        if typ=='State':
            rest=re.sub(r'^\\Return',r'\\textbf{return}',rest)
            txt=cm(rest)
        elif typ=='Return': txt='\\textbf{return} '+cm(rest)
        elif typ=='If':
            cond=re.match(r'\{(.*)\}(.*)',rest)
            txt='\\textbf{if} '+cond.group(1)+' \\textbf{then}'+cm(cond.group(2))
        elif typ=='Else': txt='\\textbf{else}'
        elif typ=='EndIf': txt='\\textbf{end if}'
        out.append(f'{n}. {ind}{txt}\n')
        if typ in('If','Else'): depth+=1
    return '\n'.join(out)
s=re.sub(r'\\begin\{algorithm\*\}(?:\[[^\]]*\])?(.*?)\\end\{algorithm\*\}',algo,s,flags=re.S)

s=re.sub(r'\\textsc\{([^}]*)\}',lambda m:'\\mathrm{'+m.group(1).replace('\\_','\\_')+'}',s)
# ---- tables: simplify
def _makecell(t):
    out=[];i=0
    while True:
        j=t.find('\\makecell',i)
        if j<0: out.append(t[i:]); break
        out.append(t[i:j]); k=j+len('\\makecell')
        if t[k]=='[': k=t.index(']',k)+1
        assert t[k]=='{'; depth=0; m=k
        while True:
            if t[m]=='{': depth+=1
            elif t[m]=='}':
                depth-=1
                if depth==0: break
            m+=1
        out.append(t[k+1:m].replace('\\\\',' ')); i=m+1
    return ''.join(out)
s=_makecell(s)
s=re.sub(r'\\cmidrule(\([^)]*\))?\{[^}]*\}','',s)
s=s.replace('\\renewcommand{\\arraystretch}{1.15}','').replace('\\renewcommand{\\arraystretch}{1.1}','').replace('\\renewcommand{\\arraystretch}{1.05}','')
s=re.sub(r'\\setlength\{\\tabcolsep\}\{[^}]*\}','',s)
s=re.sub(r'L\{([^}]*)\}',r'p{\1}',s)
s=re.sub(r'\\parbox\{[^}]*\}\{(.*?)\}\n\\end\{table',r'\1\n\\end{table',s,flags=re.S)
s=re.sub(r'\\par\\smallskip','',s)
s=re.sub(r'\\(footnotesize|scriptsize|small)\b','',s)
# nomenclature: turn heading table into table with caption-like first line
s=s.replace('\\textbf{Nomenclature}\\\\[2pt]','\\caption{Nomenclature}')
s=s.replace('\\\\[2pt]','\\\\')
# paragraphs run-in heads
s=re.sub(r'\\paragraph\{([^}]*)\}',r'\\textit{\1.}',s)
# frontmatter
s=re.sub(r'\\begin\{frontmatter\}','',s); s=re.sub(r'\\end\{frontmatter\}','',s)
s=re.sub(r'\\begin\{keyword\}(.*?)\\end\{keyword\}',lambda m:'\\noindent\\textit{Keywords.} '+m.group(1).replace(' \\sep',',').strip()+'\n',s,flags=re.S)
s=s.replace('\\linenumbers','')
s=re.sub(r'\\journal\{[^}]*\}','',s)
# preamble: replace with simple
s=re.sub(r'^.*?\\begin\{document\}',lambda m:'\\documentclass{article}\n\\usepackage{amsmath,amssymb,amsthm,graphicx}\n\\newtheorem{proposition}{Proposition}\n\\newtheorem{corollary}{Corollary}\n'+macros+'\n\\begin{document}',s,count=1,flags=re.S)
s=s.replace('\\title{','\\title{').replace('\\begin{abstract}','\\maketitle\n\\begin{abstract}',1)

# ---- docx-specific normalizations
s=re.sub(r'\\begin\{(table|figure)\*\}',r'\\begin{\1}',s)
s=re.sub(r'\\end\{(table|figure)\*\}',r'\\end{\1}',s)
s=s.replace('@{}','')
s=s.replace(r'\newcommand{\sbar}{\bar{\sigma}_0}',r'\newcommand{\sbar}{\overline{\sigma}_0}')
s=s.replace(r'\qquad (',r'\qquad\qquad (')
s=re.sub(r'\\noindent (\d+): ',r'\1: ',s)

open(OUT,'w').write(s)
print('ok',len(s))
