VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip


$(VENV_DIR)/bin/activate:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

.PHONY: install
install: $(VENV_DIR)/bin/activate
	$(PIP) install -r requirements.txt

.PHONY: gen
gen: openai_code.py
	$(PYTHON) openai_code.py
	# $(PYTHON) extract_html.py

# .PHONY: eval
# eval: clip_score.py render_imgs.py parquet_to_img.py
# 	$(PYTHON) parquet_to_img.py
# 	$(PYTHON) render_imgs.py
# 	$(PYTHON) clip_score.py
