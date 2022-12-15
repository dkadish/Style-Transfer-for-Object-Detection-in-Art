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
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(t)