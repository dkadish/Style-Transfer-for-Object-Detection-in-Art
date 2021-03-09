from zipfile import ZipFile
from tqdm import tqdm
from pathlib import Path
import tarfile


def unzip(zf, path=None, overwrite_existing=False):
    with ZipFile(zf) as z:
        for n in tqdm(iterable=z.namelist(), total=len(z.namelist())):
            # print(f)
            f = Path(n)
            if not f.exists() or overwrite_existing:
                try:
                    z.extract(str(f), path=path)
                except KeyError as e:
                    print(e)

def tar(tf, directory: Path, show_progress=True):
    with tarfile.open(tf, "w:xz") as t:
        filelist = list(directory.glob('**/*'))
        if show_progress:
          for f in tqdm(filelist, total=len(filelist)):
              t.add(f, arcname=f.relative_to(directory.parent))
        else:
          for f in filelist:
              t.add(f, arcname=f.relative_to(directory.parent))

def untar(tf, directory: Path = Path('.'), show_progress=True):
    with tarfile.open(tf, "r:xz") as t:
        t.extractall()