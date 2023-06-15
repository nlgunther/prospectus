import numpy as np, re, PyPDF2 as ppdf2, pickle as pkl, os
from tqdm import tqdm
text = 'ablankcblankde'
rex = re.compile('blank')

# get files from Downloads folder
from_downloads = lambda fn: os.path.join("..\..\..\Downloads",fn)
sample_cas_fn = from_downloads('sample_cas.pkl')

read_from_pdf = False

if read_from_pdf:

    reader = ppdf2.PdfReader(from_downloads("connave-2016-c01-prospectus(1).pdf"))
    pages = [reader.pages[i].extract_text() for i in tqdm(range(reader.numPages))]

    pkl.dump(pages,open(sample_cas_fn,'wb'))

else:
    
    assert os.path.isfile(sample_cas_fn)
    pages = pkl.load(open(sample_cas_fn,'rb'))
# print(pages[25])
isx = lambda x: x[1]
rex = 'Allocation of Senior Reduction Amount and Subordinate Reduction Amount'

class TextMarker(object): 
    
    @staticmethod
    def forfilter(f): return lambda x: f(x)
    
    second = staticmethod(lambda x: x[1])

    @staticmethod
    def get_match(text,rex): return list(re.finditer(rex,text))
    
    @staticmethod
    def mark_(match, text, marker):
        start = match.start()
        return text[:start]+marker+text[start:]
    
    @staticmethod
    def mark(match,text_,start_marker=None,end_marker=None):
        text = text_[:]
        for marker in (start_marker,end_marker):
            if marker: text = mark_(match,text,marker)
        return text  

    @staticmethod
    def insp_match_(text,match,margin=100,upper=True):
        matchtext = match.group()
        if upper: matchtext = matchtext.upper()
        return text[max(0,match.start()-margin):match.start()]+\
                    match.group().upper() +\
                    text[match.end():min(len(text),match.end() + margin)]
    
    @classmethod
    def get_matches_(kls,texts,rex): return list(filter(kls.second,[(i,kls.get_match(text,rex))
                                                                    for i,text in enumerate(texts)]))
   
    @classmethod
    def insp_match(kls,text,matches,margin=100): return [kls.insp_match_(text,match,margin) for match in matches]

    
    def __init__(self,start_marker,end_marker,pages):
        self.start_marker = start_marker
        self.end_marker = end_marker
        self.pages = pages
        
    def get_matches(self,rex):
        self.matches = self.get_matches_(self.pages,rex)
        return len(self.matches)
    
    
    def mark(self,rex,matches_num,
             match_num = 0,
             start=None,
             end=None,
             text = None):
        print(matches_num)
        pagenum,matches = self.matches[matches_num]
        text = text if text else self.pages[pagenum][:]
        match = matches[match_num]
#         print('match',match)
        markers = filter(self.__class__.second,zip((self.start_marker,self.end_marker),(start,end)))
        for marker,_ in markers: 
            text = self.mark_(match,text,marker)
        return text
        
    def excerpt(self,margin=100):
        return [(i,matches_[0], self.insp_match(self.pages[matches_[0]],matches_[1],margin=margin))
                for i, matches_ in enumerate(self.matches)]
        
    def display(self):
            for i,j,excerpt in self.excerpt():
                print('\nmatched_page_number %i, page number %i, number of matches on page %i\n-----------------\n' % (i,j,len(excerpt)))
                print('\n###\n'.join(excerpt))

relpages_ = filter(isx,[(i,TextMarker.get_matches_(page,rex)) for i,page in enumerate(pages)])
tm = TextMarker('START_','END_',pages)

relpages = (list(relpages_))
insp_match_ = lambda text,match,margin=100: text[max(0,match.start()-margin):min(len(text),match.end() + margin)]
insp_match = lambda text,matches,margin=100: [insp_match_(text,match,margin) for match in matches]

display_replages = False

if display_replages:
    for i, matches in relpages:
        print('\n%i\n-----------------\n' % i)
        print(i,'\n#####\n'.join(insp_match(pages[i],matches)))


assert 2 == TextMarker.second((1,2))

rexar = 'Allocation of Senior Reduction Amount and Subordinate Reduction Amount;Allocation of Tranche Write.*down Amounts;Allocation of Tranche Write.*up Amounts'.split(';')
rexs = '|'.join(rexar)

tm = TextMarker(*['_%s:ALLOCATION' %s for s in 'START END'.split()],pages = pages)

tm.get_matches(rexs)
tm.display()

# print('\n\n'.join(map(str,tm.excerpt(200))))
if False: print('\n\n'.join(map(str,
                      filter(lambda x:  re.search('(?i)On each Payment Date',
                                ' '.join(x[2]).replace('\n',' ')),
                                 tm.excerpt(200)))))

