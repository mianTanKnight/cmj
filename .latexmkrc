# Force latexmk to use XeLaTeX even when it thinks it is running "pdflatex"
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$pdf_mode = 1;

# Optional: put outputs into ./out (match LaTeX Workshop outDir)
$out_dir = 'out';
