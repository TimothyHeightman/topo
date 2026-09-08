LATEXMK = env LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error
PYTHON ?= python3

.PHONY: figures figure1 density-comparison figure2 density-lift supp

all-figures: figure1 density-comparison figure2 density-lift supp

figures:
	$(PYTHON) experiments/plot_all.py

figure1:
	cp experiments/results/one_qubit_fnqs/log_infidelity_sphere_zoom_t241_p481.pdf figures/figure1_obstruction.pdf
	cd figures && $(LATEXMK) figure1_static.tex

density-comparison:
	$(PYTHON) experiments/plot_density_lift_comparison.py

figure2:
	$(PYTHON) experiments/plot_figure2_dynamics.py

density-lift:
	$(PYTHON) experiments/plot_figure3_sixpanel.py

supp:
	$(PYTHON) experiments/plot_supp_susceptibility.py
	$(PYTHON) experiments/plot_supp_pairs.py
	$(PYTHON) experiments/plot_supp_throttling.py
