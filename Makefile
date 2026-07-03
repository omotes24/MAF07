.PHONY: setup verify splits features closed ood_fair ood_oracle ablation aggregate tables figures audit all test

setup:
	bash scripts/setup_env.sh

verify:
	bash scripts/verify_dataset.sh

splits:
	bash scripts/make_splits.sh

features:
	bash scripts/extract_features.sh

closed:
	bash scripts/run_closed.sh

ood_fair:
	bash scripts/run_ood_fair.sh

ood_oracle:
	bash scripts/run_ood_oracle.sh

ablation:
	bash scripts/run_ablation.sh

aggregate:
	bash scripts/aggregate.sh

tables:
	bash scripts/make_tables.sh

figures:
	bash scripts/make_figures.sh

audit:
	maf07 audit-coverage --strict

all:
	bash scripts/run_all.sh --resume --workers auto

test:
	pytest -q

