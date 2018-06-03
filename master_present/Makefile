%.tex: %.md preamble.yml preamble.tex guillemots.hs
	pandoc $*.md preamble.yml --include-in-header=preamble.tex -t beamer -o $*.tex

%.pdf: %.tex
	xelatex $*.tex
	xelatex $*.tex

all: slides.pdf 

.PHONY: all

clean: 
	rm *.pdf 
	rm *.toc 
	rm *.nav 
	rm *.log
	rm *.aux 
	rm *.snm
	rm *.out

