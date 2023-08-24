run: train.py model.py baselines.py tasks.py
	python3 train.py

cts:
	ls *.py | entr make

deps: requirements.txt
	pip install -r requirements.txt

test:
	pytest

clean:
	rm -rf __pycache__
