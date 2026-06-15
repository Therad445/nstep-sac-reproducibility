.PHONY: check pendulum plots

check:
	bash scripts/check_project.sh

pendulum:
	bash scripts/run_pendulum_ablation.sh

plots:
	bash scripts/make_plots.sh
