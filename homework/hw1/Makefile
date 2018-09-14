PROJECT_NAME=hw1
SRC=$(wildcard *.tex *.sty)

.PHONY: $(PROJECT_NAME).pdf all clean cleanall view

all: $(PROJECT_NAME).pdf $(SRC)

$(PROJECT_NAME).pdf: $(PROJECT_NAME).tex
	# Alternatively, use batchmode to elliminate console spam
	# latexmk -pdf --synctex=1 -file-line-error -interaction=batchmode -shell-escape $<
	latexmk -pdf --synctex=1 -file-line-error -interaction=nonstopmode -shell-escape $<

# A target to build with better error messages when latexmk craps itself.
error:
	pdflatex -shell-escape -file-line-error $(PROJECT_NAME).tex

clean:
	latexmk -c
	rm -rf _minted-*
	rm -f *.synctex\(busy\)

cleanall:
	latexmk -C
	rm -rf _minted-*
	rm -f *.synctex\(busy\)
	rm -f *.synctex.gz *.bbl *.nav *.snm *.vrb *.tdo

view:
	1>/dev/null 2>/dev/null xdg-open $(PROJECT_NAME).pdf &
