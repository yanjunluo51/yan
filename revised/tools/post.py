import zipfile,re,sys,shutil
src,dst=sys.argv[1],sys.argv[2]
zin=zipfile.ZipFile(src)
zout=zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data=zin.read(it.filename)
    if it.filename=='word/document.xml':
        s=data.decode('utf8')
        s=re.sub(r'(<m:t[^>]*>)([^<]*)(</m:t>)',lambda m:m.group(1)+m.group(2).replace('*','∗')+m.group(3),s)
        s=s.replace('<m:nor /><m:sty m:val="p" />','<m:nor />').replace('<m:nor/><m:sty m:val="p"/>','<m:nor/>')
        data=s.encode('utf8')
    if it.filename=='[Content_Types].xml':
        s=data.decode('utf8')
        if 'Extension="png"' not in s:
            s=s.replace('<Default Extension="xml"','<Default Extension="png" ContentType="image/png" /><Default Extension="xml"',1)
        data=s.encode('utf8')
    zout.writestr(it,data)
zout.close(); print('post ok')
