# Build the paper. Requires a TeX distribution with booktabs and pgfplots
# (texlive-latex-recommended + texlive-latex-extra + texlive-pictures on Debian/Ubuntu).
#
#   make          -> main.pdf
#   make assets   -> regenerate tables/ and figs/ from the repo's CSV and JSON
#   make arxiv    -> arxiv.tar.gz, ready to upload
#   make clean    -> remove build intermediates (keeps main.pdf)
#   make distclean-> also remove main.pdf and the tarball

TEX    := pdflatex -interaction=nonstopmode -halt-on-error
PYTHON := python

.PHONY: all assets arxiv clean distclean check

all: main.pdf

# Three passes: the first resolves \input and writes .aux, bibtex reads it, the last two
# settle references and the bibliography.
main.pdf: main.tex refs.bib $(wildcard tables/*.tex) $(wildcard figs/*.tex)
	$(TEX) main.tex
	-bibtex main
	$(TEX) main.tex
	$(TEX) main.tex

assets:
	$(PYTHON) ../paper/scripts/make_paper_assets.py

# Fails loudly if anything is unresolved -- easy to miss in pdflatex's output.
check: main.pdf
	@! grep -q 'Undefined control sequence' main.log || { echo 'FAIL: undefined control sequence'; exit 1; }
	@! grep -qE '(There were undefined references|Citation .* undefined|LaTeX Warning: Reference)' main.log \
		|| { echo 'FAIL: undefined reference or citation'; exit 1; }
	@echo 'OK: no undefined references or citations'
	@grep -c 'Overfull' main.log | xargs -I{} echo 'overfull boxes: {}'

# arXiv wants the source, not the intermediates, and it runs bibtex itself only if the
# .bbl is absent -- shipping the .bbl is the reliable route.
arxiv: main.pdf
	@test -f main.bbl || { echo 'main.bbl missing; run make first'; exit 1; }
	tar czf arxiv.tar.gz main.tex main.bbl refs.bib tables figs
	@echo 'wrote arxiv.tar.gz:'
	@tar tzf arxiv.tar.gz

clean:
	rm -f main.aux main.log main.out main.bbl main.blg main.toc

distclean: clean
	rm -f main.pdf arxiv.tar.gz
