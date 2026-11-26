"""Refresh interpretation cells from frozen outputs without refitting or rescoring test."""
from pathlib import Path
import json
import runpy
import nbformat
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, Markdown
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'/'next_song'
old=nbformat.read(ROOT/'assignment_2.ipynb',as_version=4)
new=runpy.run_path(str(ROOT/'build_continuation_notebook.py'),run_name='definitions_only')['notebook']
assert len(old.cells)==len(new.cells)
summary=json.loads((OUT/'experiment_summary.json').read_text())
table=pd.read_csv(OUT/'test_metrics.csv')
namespace={'np':np,'pd':pd,'display':display,'Markdown':Markdown,'OUT':OUT,
           'summary':summary,'primary':summary['primary'],'table':table,
           'MODEL_FEATURES':{'History':None,'History + artist':None,'History + artist + audio':None},
           'results':{'details':pd.read_csv(OUT/'test_per_query.csv'),
                      'sensitivity':pd.read_csv(OUT/'sensitivity_metrics.csv'),
                      'subgroup':pd.read_csv(OUT/'test_support_metrics.csv')}}
InteractiveShell.instance()
for before,after in zip(old.cells,new.cells):
    if after.cell_type=='markdown':
        continue
    after.execution_count=before.execution_count
    after.outputs=before.outputs
    should_execute=any(marker in after.source for marker in
                       ['**Result in this sample.**','sensitivity = results[','### 5.2 What'])
    if should_execute:
        with capture_output() as captured:
            exec(after.source,namespace)
        outputs=[]
        if captured.stdout:
            outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        outputs.extend(nbformat.v4.new_output('display_data',data=o.data,metadata=o.metadata)
                       for o in captured.outputs)
        after.outputs=outputs
nbformat.validate(new)
nbformat.write(new,ROOT/'assignment_2.ipynb')
print('Re-executed three interpretation cells against frozen outputs; no models or test rankings rerun.')
