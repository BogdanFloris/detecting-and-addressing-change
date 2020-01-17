.PHONY: lint, test, clean

XARGS := xargs -0 $(shell test $$(uname) = Linux && echo -r)

lint:
	flake8 --show-source .

test:
	pytest --cov=zetamanager --cov-report term-missing tests

clean:
	find . \( -name '*.py[co]' -o -name dropin.cache \) -print0 | $(XARGS) rm
	rm .coverage
