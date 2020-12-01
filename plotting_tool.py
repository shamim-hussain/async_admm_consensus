import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pathlib import Path
import sys


if __name__ == '__main__':
    dataframes={}
    folder = Path(sys.argv[1])
    
    subfolders = list(f for f in folder.iterdir() if f.name.lower()!='figures')
    
    for sf in subfolders:
        dataframes[sf.name]=pd.read_pickle(str(sf/'logs.pkl'))


breakname = lambda name: ' '.join(v[0].upper()+v[1:].lower()
                                  for v in name.split('_'))


matplotlib.use('agg')

xlim = {'wall_time':1,'global_step':200}
figs = {}
figsize=[4,3]
for k, df in dataframes.items():
    for xl in df.columns[:2]:
        for yl in df.columns[2:]:
            if (xl,yl) not in figs:
                figs[(xl,yl)] = plt.figure(figsize=figsize)
                
            fig = figs[(xl,yl)]
            ax = fig.gca()
            ax.title.set_text(f'{breakname(yl)} vs. {breakname(xl)}')
            ax.plot(df[xl],df[yl], label=k)
            ax.set_xlim(right=xlim[xl])
            ax.legend()
            ax.grid(True)            


figdir = folder/'Figures'
figdir.mkdir(exist_ok=True)
for (xl, yl), f in figs.items():
    f.savefig(figdir/f'{breakname(yl)} vs. {breakname(xl)}.png')
    f.savefig(figdir/f'{breakname(yl)} vs. {breakname(xl)}.pdf')