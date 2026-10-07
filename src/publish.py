"""Build static figures, an offline research viewer and the Russian PDF report."""
import json
import re
from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

from .pipeline import SHARES

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#178579', '#5664bd', '#bf6b25']
LABELS = ['C0. Базовая корзина', 'C1. Смешанный профиль', 'C2. Высокие расходы и услуги']
MONTHS = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']


def figures(arr, profiles, quality, stability):
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#1b3140', 'text.color': '#1b3140',
                         'axes.edgecolor': '#b8c4ca', 'savefig.facecolor': 'white'})
    def save(name, fig):
        fig.savefig(out / (name + '.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.7))
    for c in range(3):
        mask = arr['labels'][-1] == c
        ax.scatter(*arr['xy'][-1, mask].T, s=9, alpha=.52, c=COLORS[c], label=LABELS[c], linewidths=0)
    ax.set(xlabel='PC1 (39,57% дисперсии на срезах настройки)', ylabel='PC2 (16,39%)')
    ax.legend(fontsize=9, frameon=False, loc='upper left')
    ax.set_title('Экономическое пространство, декабрь 2024', loc='left', weight='bold', pad=12)
    save('economic_space', fig)
    fig, ax = plt.subplots(figsize=(10, 3.3))
    vals = profiles.query('month == "2024-12"').set_index('cluster')[[f'share_{s}' for s in SHARES]].to_numpy() * 100
    ax.imshow(vals, cmap='YlGnBu', vmin=0, vmax=50, aspect='auto')
    ax.set_xticks(range(6), ['Продукты', 'Здоровье', 'Маркетплейсы', 'Общепит', 'Транспорт', 'Прочие'])
    ax.set_yticks(range(3), ['C0', 'C1', 'C2'])
    ax.tick_params(axis='x', labelsize=8.5)
    for c in range(3):
        for j in range(6): ax.text(j, c, f'{vals[c,j]:.1f}%', ha='center', va='center', color='white' if vals[c,j]>27 else '#142b3a', fontsize=12)
    ax.set_title('Медианы долей расходов по группам, декабрь 2024', loc='left', weight='bold', pad=12)
    save('profiles', fig)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for model, color, label in [('Leiden_combined_k40_r0.4',COLORS[0],'Leiden'),('KMeans_6',COLORS[1],'K-means(6)'),('KMeans_8',COLORS[2],'K-means(8)')]:
        q=quality[quality.candidate==model].sort_values('month')
        ax.plot(range(12), q.SW, marker='o', lw=2, color=color, label=label)
    ax.axvspan(8.5,11.5,color='#eef3f4',zorder=0)
    ax.set(xticks=range(12),xticklabels=MONTHS,ylabel='Средний silhouette',ylim=(0,.26),xlim=(-.25,11.25))
    ax.legend(frameon=False,ncol=3,fontsize=9)
    ax.set_title('Качество в течение 2024 года',loc='left',weight='bold',pad=12)
    save('quality',fig)
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(10,4.8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for c in range(3):
        ax.plot(range(12),profiles[profiles.cluster==c].n,marker='o',color=COLORS[c],label=f'C{c}',lw=2)
    ar=stability.groupby('month').ARI.agg(['median','min','max'])
    ax2.fill_between(range(12),ar['min'],ar['max'],color='#d6e7e5')
    ax2.plot(range(12),ar['median'],color='#174a49',marker='o')
    ax.set(ylabel='Территорий');ax.legend(frameon=False,ncol=3)
    ax2.set(xticks=range(12),xticklabels=MONTHS,ylabel='ARI',ylim=(0,1.05))
    ax.set_title('Численность групп и чувствительность к девяти вариантам',loc='left',weight='bold',pad=12)
    ax2.annotate('Сентябрь: медиана 0,361',xy=(8,ar.iloc[8]['median']),xytext=(5,.12),fontsize=9)
    save('dynamics',fig)
    i=int(np.flatnonzero(arr['ids']==191)[0])
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(10,4.5),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    ax.plot(range(12),arr['shares'][12:,i,2]*100,lw=2.3,marker='o',color=COLORS[0],label='Маркетплейсы')
    ax.plot(range(12),arr['shares'][12:,i,0]*100,lw=2.0,marker='o',color=COLORS[1],label='Продовольствие')
    ax.set(ylabel='Доля расходов, %');ax.legend(frameon=False,ncol=2)
    ax2.scatter(range(12),arr['labels'][:,i],c=[COLORS[x] for x in arr['labels'][:,i]],s=arr['support'][:,i]*150+10,zorder=3)
    ax2.plot(range(12),arr['labels'][:,i],color='#bac4ca',lw=1)
    ax2.set(xticks=range(12),xticklabels=MONTHS,yticks=[0,1,2],yticklabels=['C0','C1','C2'],ylim=(-.4,2.3),ylabel='Группа')
    ax.set_title('Юстинский район (ID 191): переход C1 в C0 в июле',loc='left',weight='bold',pad=12)
    ax2.text(.01,.95,'Размер точки отражает поддержку',transform=ax2.transAxes,va='top',fontsize=8)
    save('case191',fig)
    edges=pd.read_csv(ROOT/'results/economic_edges_december.csv');d=edges.road_km.replace([np.inf,-np.inf],np.nan).dropna()
    bins=[0,250,500,1000,2000,4000,np.inf];hist=np.histogram(d,bins=bins)[0]
    fig,ax=plt.subplots(figsize=(10,3.6))
    ax.bar(['0-250','250-500','500-1000','1000-2000','2000-4000','>4000'],hist/len(d)*100,color=COLORS[0],width=.65)
    ax.set(xlabel='Дорожное расстояние между центрами, км',ylabel='Доля рёбер, %')
    ax.set_title('Дорожные расстояния связей экономической сети',loc='left',weight='bold',pad=12)
    save('distances',fig)


def make_report():
    font=font_manager.findfont('DejaVu Sans')
    bold=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))
    pdfmetrics.registerFont(TTFont('DV',font));pdfmetrics.registerFont(TTFont('DV-Bold',bold))
    pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
    styles={
      'body':ParagraphStyle('body',fontName='DV',fontSize=9.5,leading=14.1,spaceAfter=9,textColor=colors.HexColor('#1c3342')),
      'title':ParagraphStyle('title',fontName='DV-Bold',fontSize=28,leading=35,spaceAfter=22,textColor=colors.HexColor('#164b48')),
      'h':ParagraphStyle('h',fontName='DV-Bold',fontSize=18,leading=24,spaceAfter=17,textColor=colors.HexColor('#164b48')),
      'cell':ParagraphStyle('cell',fontName='DV',fontSize=8,leading=11,textColor=colors.HexColor('#1c3342'))}
    doc=SimpleDocTemplate(str(ROOT/'reports/Report.pdf'),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=43,bottomMargin=43,title='Муниципальные траектории',author='Муниципальные траектории')
    story=[]; text=(ROOT/'reports/Report.md').read_text();blocks=text.split('\n\n')
    for b in blocks:
        b=b.strip()
        if not b:continue
        if b.startswith('# '):story.append(Spacer(1,100));story.append(Paragraph(escape(b[2:]),styles['title']))
        elif b.startswith('## '):story.append(PageBreak());story.append(Paragraph(escape(b[3:]),styles['h']))
        elif b.startswith('!['):
            target=re.search(r'\]\((.*?)\)',b).group(1)
            img=Image(str((ROOT/'reports'/target).resolve()));ratio=img.imageHeight/img.imageWidth
            width = min(doc.width, 225/ratio) if 'dynamics' in target else doc.width
            story.append(Spacer(1,4));story.append(Image(img.filename,width=width,height=width*ratio))
        elif b.startswith('|'):
            rows=[x.strip().strip('|').split('|') for x in b.splitlines() if not re.match(r'^\|[\s:|-]+$',x)]
            rows=[[Paragraph(escape(x.strip()),styles['cell']) for x in row] for row in rows]
            n=len(rows[0]); widths=[doc.width/n]*n
            if n==8:widths=[doc.width*.27]+[doc.width*.73/7]*7
            elif n==6:widths=[doc.width*.25]+[doc.width*.75/5]*5
            elif n==3:widths=[doc.width*.15,doc.width*.19,doc.width*.66]
            t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2efed')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#9fb8b5')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f6f7')])]))
            story.extend([t,Spacer(1,12)])
        else:story.append(Paragraph(escape(b).replace('\n',' '),styles['body']))
    def footer(canvas,doc):
        canvas.setFont('DV',7);canvas.setFillColor(colors.HexColor('#667b85'))
        canvas.drawString(42,24,'Муниципальные траектории · СберИндекс · данные 2023-2024')
        canvas.drawRightString(A4[0]-42,24,str(doc.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def make_viewer(arr,quality,profiles):
    names=pd.read_csv(ROOT/'data/labels/municipal_names.csv').fillna('').set_index('territory_id')
    peers=pd.read_csv(ROOT/'results/peers_december.csv')
    events=pd.read_csv(ROOT/'results/transition_events.csv')
    members=pd.read_csv(ROOT/'results/memberships.csv')
    records=[]
    for i,tid in enumerate(arr['ids']):
        q=names.loc[int(tid)];p=peers[peers.territory_id==tid]
        records.append({'id':int(tid),'name':q['name'],'region':q.region,'type':q['type'],
                        'xy':np.round(arr['xy'][:,i],4).tolist(),'c':arr['labels'][:,i].tolist(),
                        's':np.round(arr['support'][:,i],4).tolist(),
                        'sh':np.round(arr['shares'][12:,i]*100,3).tolist(),
                        'total':arr['levels'][12:,i,0].astype(int).tolist(),
                        'yoy':np.round((arr['levels'][12:,i,0]/arr['levels'][:12,i,0]-1)*100,2).tolist(),
                        'sil':np.round(members[members.territory_id==tid].silhouette,3).tolist(),
                        'peers':[[int(r.peer_id),round(r.feature_distance,3),None if pd.isna(r.road_km) else float(r.road_km)] for r in p.itertuples()]})
    ev=events[events.robust].sort_values('feature_displacement',ascending=False)
    er=[[int(r.territory_id),r.month,int(r.from_cluster),int(r.to_cluster),round(r.support_min,3),round(r.feature_displacement,3)] for r in ev.itertuples()]
    payload=json.dumps({'rows':records,'events':er,'quality':quality.replace([np.inf,-np.inf],None).to_dict('records')},ensure_ascii=False,separators=(',',':')).replace('</',r'<\/')
    template=(ROOT/'src/viewer_template.html').read_text()
    (ROOT/'site').mkdir(exist_ok=True)
    (ROOT/'site/index.html').write_text(template.replace('/*__DATA__*/',payload),encoding='utf-8')


def main():
    arr=np.load(ROOT/'results/analysis_arrays.npz')
    profiles=pd.read_csv(ROOT/'results/cluster_profiles.csv');quality=pd.read_csv(ROOT/'results/monthly_quality.csv');stability=pd.read_csv(ROOT/'results/stability.csv')
    ev=pd.read_csv(ROOT/'results/transition_events.csv')
    pd.DataFrame([{'min_support':s,'min_displacement':d,'events':int((ev.persistent_two_months&(ev.support_min>=s)&(ev.feature_displacement>=d)).sum())} for s in [.7,.8,.9] for d in [.25,.5,.75]]).to_csv(ROOT/'results/threshold_sensitivity.csv',index=False)
    figures(arr,profiles,quality,stability)
    make_report();make_viewer(arr,quality,profiles)
    print('Built figures, reports/Report.pdf and site/index.html')


if __name__=='__main__':main()
