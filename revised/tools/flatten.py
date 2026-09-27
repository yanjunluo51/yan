import re,os
os.chdir('/home/user/yan/revised')
def expand(path):
    s=open(path).read()
    def rep(m):
        p=m.group(1)
        if not p.endswith('.tex'): p+='.tex'
        return expand(p)
    return re.sub(r'\\input\{([^}]+)\}',rep,s)
out=expand('CAIE_main_revised_src.tex')
open('CAIE_main_revised.tex','w').write(out)
print(len(out))
