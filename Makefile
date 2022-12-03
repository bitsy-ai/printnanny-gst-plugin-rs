.PHONY: patch minor major patch-execute minor-execute major-execute

DEV_USER ?= pi
DEV_MACHINE ?= pn-dev

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

dev-build:
	cross build --workspace --target=aarch64-unknown-linux-gnu
	rsync --progress -e "ssh -o StrictHostKeyChecking=no" target/aarch64-unknown-linux-gnu/debug/printnanny-gst-pipeline $(DEV_USER)@$(DEV_MACHINE).local:~/printnanny-gst-pipeline

