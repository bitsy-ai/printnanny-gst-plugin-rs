.PHONY: patch minor major patch-execute minor-execute major-execute

patch:
	cargo release patch --no-verify --no-publish

patch-execute:
	cargo release patch --no-verify --no-publish --execute

minor:
	cargo release minor--no-verify --no-publish
	
minor-execute:
	cargo release minor --no-verify --no-publish --execute

major:
	cargo release major -no-verify --no-publish
	
major-execute:
	cargo release major --no-verify --no-publish --execute
