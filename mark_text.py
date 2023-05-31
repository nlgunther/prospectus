import numpy as np, re, PyPDF2 as ppdf2, pickle as pkl, os
from tqdm import tqdm
text = 'ablankcblankde'
rex = re.compile('blank')
get_matches = lambda text,rex: list(re.finditer(rex,text))


def mark_(match, text, marker):
    start = match.start()
    return text[:start]+marker+text[start:]
matches = get_matches(text,rex)
def mark(match,text_,start_marker=None,end_marker=None):
    text = text_[:]
    for marker in (start_marker,end_marker):
        if marker: text = mark_(match,text,marker)
    return text
out = mark(matches[1],text,'_START_','_END_')
out = mark(matches[1],text,end_marker='_END_')
out = mark(matches[1],text,'_START_')

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
relpages_ = filter(isx,[(i,get_matches(page,rex)) for i,page in enumerate(pages)])
relpages = (list(relpages_))
insp_match_ = lambda text,match,margin=100: text[max(0,match.start()-margin):min(len(text),match.end() + margin)]
insp_match = lambda text,matches,margin=100: [insp_match_(text,match,margin) for match in matches]

display_replages = False

if display_replages:
    for i, matches in relpages:
        print('\n%i\n-----------------\n' % i)
        print(i,'\n#####\n'.join(insp_match(pages[i],matches)))
