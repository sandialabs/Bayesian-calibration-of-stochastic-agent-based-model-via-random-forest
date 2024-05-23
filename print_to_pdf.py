from glob import glob
from pypdf import PdfMerger
import os
import shutil
import subprocess

convert_map = {
    ".py": "python",
    ".R": "r"
}
ignore_exts = [".csv", ".json", ".pkl", ".mplstyle", "", ".txt", ".yml", ".png"]

if os.path.isdir("docs"):
    shutil.rmtree("docs")

# Write source files, md, pdf to docs/
for path in glob("**", recursive=True):
    if not os.path.isdir("docs"):
        os.mkdir("docs")
    if os.path.isfile(path):
        _,ext = os.path.splitext(path)
        if ext not in ignore_exts:
            if ext == ".pdf":
                new_name = path.replace("/", "_")
                shutil.copyfile(path, "docs/"+new_name)
            elif ext == ".md":
                # pdf_name = "docs/"+path.replace("/", "_").replace(ext, ".pdf")
                # args = ["pandoc", "--pdf-engine=tectonic", path, "-o", pdf_name]
                pdf_name = path.replace("/", "_").replace(ext, ".pdf")
                args = ["quarto", "render", path, "--to", "pdf", "--output-dir", "docs/", "-o", pdf_name, "--quiet"]
                subprocess.check_call(args)
            else:
                md_name = "docs/"+path.replace("/", "_").replace(ext, ".md")
                pdf_name = path.replace("/", "_").replace(ext, ".pdf")
                with open(md_name, "w") as f1:
                    f1.write(path+"\n\n")
                    f1.write("```"+convert_map[ext]+"\n")
                    with open(path, "r") as f2:
                        lines = f2.readlines()
                        f1.writelines(lines)
                    f1.write("```")
                # args = ["pandoc", "--pdf-engine=tectonic", md_name, "-o", pdf_name]
                args = ["quarto", "render", md_name, "--to", "pdf", "--output-dir", "docs/", "-o", pdf_name, "--quiet"]
                subprocess.check_call(args)
                os.remove(md_name)

# Merge pdfs
pdfs = sorted(glob("docs/*.pdf"))
plot_pdfs = [p for p in pdfs if "docs/plots_" in p]
non_plot_pdfs = [p for p in pdfs if "docs/plots_" not in p]
all_pdfs = non_plot_pdfs + plot_pdfs

merger = PdfMerger()

for pdf in all_pdfs:
    merger.append(pdf)

merger.write("docs/docs.pdf")

# Clean up
for pdf in all_pdfs:
    os.remove(pdf)
# %%

