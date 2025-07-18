VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip


$(VENV_DIR)/bin/activate:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

.PHONY: install
install: $(VENV_DIR)/bin/activate
	$(PIP) install -r requirements.txt

# OPENAIにコード生成をさせる
.PHONY: gen
gen: generate_response/openai_code.py
	$(PYTHON) generate_response/openai_code.py
	$(PYTHON) extract_html.py

# 生成物の評価
.PHONY: eval
eval: calculate_similarity/clip_score.py calculate_similarity/render_img.py load_label_img.py
	$(PYTHON) calculate_similarity/render_img.py
	$(PYTHON) load_label_img.py
	$(PYTHON) calculate_similarity/clip_score.py
