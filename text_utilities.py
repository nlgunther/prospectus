import os, re
from functools import reduce
from importlib import reload
from datetime import datetime

# FILE CONFIGURATION

gdrive = r'G:\My Drive'
gfld = r'NLP\prospectus'
fld = os.path.join(gdrive,gfld)
print(fld)

# USEFUL UTILITIES

listify = lambda f: lambda *args,**kwargs: list(f(*args,**kwargs))
mapl = listify(map)
filterl = listify(filter)
reducel = listify(reduce)
mapl(str,range(10))
###############
get_methods = lambda obj: [e for e in dir(obj) if not e[0]=='_']
find_lines = lambda rex,lines,i=None: [l for l in lines if re.search(rex,l[i] if i else l)]
def xgetd(d,args):
    k,args = args[0],args[1:]
    d = d[k]
    if args: return xgetd(d,args)
    else: return d
def find_block_boundary(regexp, ls):
    return filterl(lambda l: re.search(regexp,l[1]) and not len(re.findall('\W',l[1]))>10,enumerate(ls))
beginsection = '^Hypothetical Structure and Calculations with Respect to the Reference Tranches$'
endsection = '^THE AGREEMENTS$;^The Reference Pool$'
def find_starts_ends(ls):
    starts = [tpl for begin in beginsection.split(';') for tpl in find_block_boundary(begin,ls)]
    ends = [tpl for end in endsection.split(';') for tpl in find_block_boundary(end,ls)]
    return starts,ends
def find_textblocks(starts,ends):
    ar = []
    for pair in product(starts, ends):
        # print(pair)
        if (dist:= pair[1][0] - pair[0][0]) > 0:
            ar.append((dist,pair))
        ar.sort()
    return ar
def pairs2dict(pairs):
    ar = []
    for p_ in pairs:
        dist, pair = p_
        ar.append(dict(zip('startstop tags'.split(),zip(*pair))))
    return ar
viewi = lambda span, i,labels,width=5: labels[span[i]-width:span[i]+width]#[:10]
# test =lambda i: any([ pair[0][0] <= i < pair[0][1] for pair in ar ])
get_range_test_from_spans = lambda spans: lambda i: any([ pair[0] <= i < pair[1] for pair in spans ])

# TEXT CLEANING

breakflag = '<PBRK>'
breaktext = '\n\n+'

mark_doublebreak = lambda txt, breaktext = breaktext: re.sub(breaktext,'<PBRK>',txt)
remove_linebreak = lambda txt: re.sub('\n','',txt)
reinsert_double_as_singlebreak = lambda txt: re.sub('<PBRK>','\n',txt)

base_transforms = [mark_doublebreak,remove_linebreak,reinsert_double_as_singlebreak]

class TextPipe(object):

    def __init__(self,text,transforms=list()):
        self.text = text
        self.transforms = transforms

    def transform(self):
        for transform in self.transforms:
            self.text = transform(self.text)
    def get_lines(self,strip=True):
        lines =  self.text.split('\n')
        return lines if not strip else [l.strip() for l in lines]

class File2LL(object):

    def __init__(self,path,
                mpl=150,
                mll = 75,
                linemap = None):
        self.path = path
        self.minParaLen = mpl
        self.maxLineLen = mll
        self.linemap = linemap

    def process(self,longlines):
        return longlines
    
    def readfile(self):
        with open(self.path,errors='replace') as f:
            lines = f.readlines()
        if self.linemap: 
            print('modifying lines')
            lines = self.linemap(lines)
        # labeled_para = mapl(lambda l: l.replace('\n','<NL>'),longlines)
        return lines
        
# DATE

today_ = datetime.today().date()
today = str(today_).replace('-','_')
today