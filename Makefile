# Configuration
RESULTS_DIR = results
EVAL_DIR = eval
MERGED_CSV = $(EVAL_DIR)/merged_evaluations.csv
STATS_CSV = $(EVAL_DIR)/benchmark_stats.csv
ATTENTION_DIR = $(EVAL_DIR)/attention

.PHONY: all prepare run merge analyze clean

all: prepare run merge analyze

prepare:
	yentlbench prepare

run:
	yentlbench run

results:
	yentlbench results --copy-to $(RESULTS_DIR)

merge:
	yentlbench merge --results-dir $(RESULTS_DIR) --output $(MERGED_CSV)

analyze:
	yentlbench analyze --input $(MERGED_CSV) --output-stats $(STATS_CSV) --output-attention $(ATTENTION_DIR)

clean:
	rm -rf dataset_output $(RESULTS_DIR) $(EVAL_DIR)
