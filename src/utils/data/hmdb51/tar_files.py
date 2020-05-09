import tarfile
import os

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

path = "../HMDB51_NUMPY"

for i in os.listdir(path):
    f = os.path.join(path,i)
    make_tarfile(f"{i}.tar.gz",f)

