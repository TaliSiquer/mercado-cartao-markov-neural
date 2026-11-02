"""Coleta reproduzível do IIE-Br (FGV IBRE), com cache e sem imputação.

Dependências: pandas e openpyxl. Descoberta no portal oficial; não presume
endereços de arquivos futuros. Uma falha de rede usa a última base válida.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
import shutil
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import pandas as pd

PORTAL = 'https://portalibre.fgv.br'
INDEX = PORTAL + '/taxonomy/term/97'
INITIAL = PORTAL + '/noticias/incerteza-economica-recuou-em-setembro'
FIELDS = ['iiebr', 'midia', 'expectativa']

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self._url=None; self._text=[]
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self._url=dict(attrs).get('href'); self._text=[]
    def handle_data(self, data):
        if self._url is not None: self._text.append(data)
    def handle_endtag(self, tag):
        if tag=='a' and self._url:
            self.links.append((self._url, ' '.join(self._text).strip()))
            self._url=None; self._text=[]

def get(url, timeout=20):
    if urlparse(url).scheme != 'https' or urlparse(url).hostname != 'portalibre.fgv.br':
        raise ValueError('URL fora do domínio oficial da FGV')
    req=Request(url, headers={'User-Agent':'Mozilla/5.0 (compatible; academic-IIEBr/1.0)'})
    with urlopen(req,timeout=timeout) as r:
        data=r.read(8_000_000)
    return data

def extract_links(page):
    parser=Links(); parser.feed(get(page).decode('utf-8','replace'))
    return [(urljoin(page,href),label) for href,label in parser.links]

def discover():
    # O portal lista as notícias mensais mais recentes nesta página.
    pages=[]
    for href,label in extract_links(INDEX):
        if 'incerteza' in (href+' '+label).lower() or '/noticias/' in href and 'iie' in label.lower():
            if href.startswith(PORTAL+'/') and href not in pages: pages.append(href)
    pages=[INITIAL]+[p for p in pages if p!=INITIAL]
    # Não selecionar a planilha "mais recente" só pelo nome: validar a data da série.
    results=[]
    for page in pages[:18]:
        try:
            for href,label in extract_links(page):
                if href.lower().split('?')[0].endswith('.xlsx') and 'iie' in (href+' '+label).lower():
                    results.append((href,page))
        except Exception:
            continue
    return results

def parse_xlsx(raw):
    df=pd.read_excel(io.BytesIO(raw), engine='openpyxl')
    df.columns=[str(c).strip().lower() for c in df.columns]
    if not {'data','iiebr'}.issubset(df.columns):
        raise ValueError('Planilha sem as colunas data / iiebr esperadas')
    if pd.api.types.is_numeric_dtype(df['data']):
        date=pd.to_datetime(df['data'], unit='D', origin='1899-12-30',errors='coerce')
    else:
        date=pd.to_datetime(df['data'],dayfirst=True,errors='coerce')
    out=pd.DataFrame({'data':date.dt.to_period('M').dt.to_timestamp()})
    for col in FIELDS:
        out[col]=pd.to_numeric(df[col],errors='coerce') if col in df else float('nan')
    out=out.dropna(subset=['data','iiebr']).drop_duplicates('data',keep='last').sort_values('data')
    if len(out)<24 or out['data'].max()>pd.Timestamp.today()+pd.DateOffset(months=1):
        raise ValueError('Histórico do IIE-Br inválido / com data futura')
    if (out['iiebr']<=0).any(): raise ValueError('IIE-Br contém valores não positivos')
    return out

def update(data_dir: Path, seed: Path|None=None, online=True, logger=print):
    data_dir.mkdir(parents=True,exist_ok=True)
    csv=data_dir/'fgv_iiebr_mensal.csv'
    archive=data_dir/'fgv_iiebr_arquivos'; archive.mkdir(exist_ok=True)
    candidates=[]
    if online:
        try: candidates=discover()
        except Exception as e: logger(f'FGV: descoberta indisponível ({e}); usando cache.')
    if seed is None and not csv.exists():
        for candidate in [data_dir/'iiebr_set26.xlsx',data_dir.parent/'iiebr_set26.xlsx']:
            if candidate.is_file(): seed=candidate; break
    if seed and seed.is_file(): candidates.append((str(seed),'arquivo inicial local'))
    best=None
    for url,origin in candidates:
        try:
            raw=seed.read_bytes() if url==str(seed) else get(url)
            parsed=parse_xlsx(raw)
            last=parsed['data'].max()
            if best is None or last>best[0]: best=(last,parsed,raw,url,origin)
        except Exception as e: logger(f'FGV: planilha ignorada ({url}): {e}')
    if best:
        last,df,raw,url,origin=best
        old=pd.read_csv(csv,parse_dates=['data']) if csv.exists() else pd.DataFrame(columns=['data']+FIELDS)
        if old.empty or last>=old['data'].max():
            # Preserva versões brutas, permitindo auditar revisões feitas pela FGV.
            digest=hashlib.sha256(raw).hexdigest()
            saved=archive/f'{digest[:16]}.xlsx'
            if not saved.exists(): saved.write_bytes(raw)
            df.to_csv(csv,index=False,encoding='utf-8-sig')
            manifest={'capturado_em_utc':datetime.now(timezone.utc).isoformat(),
                      'origem':url,'pagina':origin,'sha256':digest,
                      'ultimo_mes':last.strftime('%Y-%m'),'linhas':len(df)}
            (archive/'ultima_coleta.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
            logger(f'FGV: IIE-Br atualizado até {last:%Y-%m} ({len(df)} meses)')
    if not csv.exists():
        logger('FGV: sem planilha nova e sem cache; demais etapas continuam.')
        return None
    return pd.read_csv(csv,parse_dates=['data'])

def integrate(monthly, quarterly, iie):
    if iie is None or iie.empty: return monthly,quarterly
    source=iie.copy()
    source['mes_chave']=source['data'].dt.to_period('M').astype(str)
    m=monthly.copy()
    date_col='date' if 'date' in m else 'data'
    m['mes_chave']=pd.to_datetime(m[date_col]).dt.to_period('M').astype(str)
    m=m.drop(columns=[c for c in FIELDS if c in m],errors='ignore')
    m=m.merge(source[['mes_chave']+FIELDS],on='mes_chave',how='left',validate='many_to_one').drop(columns='mes_chave')
    source['tri_chave']=source['data'].dt.to_period('Q').astype(str)
    qmeans=source.groupby('tri_chave')[FIELDS].mean()
    qcount=source.groupby('tri_chave')['iiebr'].count()
    qmeans.loc[qcount<3,FIELDS]=float('nan')  # somente trimestres completos
    qmeans=qmeans.rename(columns={c:f'{c}_trimestral' for c in FIELDS}).reset_index()
    q=quarterly.copy(); dq='date' if 'date' in q else 'data'
    q['tri_chave']=pd.to_datetime(q[dq]).dt.to_period('Q').astype(str)
    q=q.drop(columns=[c for c in qmeans.columns if c!='tri_chave' and c in q],errors='ignore')
    q=q.merge(qmeans,on='tri_chave',how='left',validate='one_to_one').drop(columns='tri_chave')
    q=q.sort_values(dq)
    # Defasagem de um trimestre CALENDÁRIO, mesmo se faltarem trimestres na base.
    lag=qmeans[['tri_chave','iiebr_trimestral']].copy()
    lag['tri_chave']=(pd.PeriodIndex(lag['tri_chave'],freq='Q')+1).astype(str)
    lag=lag.rename(columns={'iiebr_trimestral':'iiebr_lag1'})
    q['tri_chave']=pd.to_datetime(q[dq]).dt.to_period('Q').astype(str)
    q=q.merge(lag,on='tri_chave',how='left',validate='one_to_one').drop(columns='tri_chave')
    return m,q

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default='Mercado_de_Cartao/data')
    p.add_argument('--seed'); p.add_argument('--offline',action='store_true')
    a=p.parse_args()
    update(Path(a.data_dir),Path(a.seed) if a.seed else None,not a.offline)
