#!/bin/bash
set -e

MDBOOK_URL="https://github.com/rust-lang/mdBook/releases/latest/download/mdbook-v0.5.3-x86_64-unknown-linux-gnu.tar.gz"

# Install mdbook if not already present
if ! command -v mdbook &>/dev/null; then
  curl -fsSL "$MDBOOK_URL" -o /tmp/mdbook.tar.gz
  tar xzf /tmp/mdbook.tar.gz -C /tmp/ mdbook
  chmod +x /tmp/mdbook
  MDBOOK=/tmp/mdbook
else
  MDBOOK=mdbook
fi

# Build each book
for book in infra ml web3; do
  echo "Building $book..."
  "$MDBOOK" build "$book"
done

# Copy output to build root
mkdir -p builds/infra builds/ml builds/web3
cp -r infra/book/* builds/infra/
cp -r ml/book/* builds/ml/
cp -r web3/book/* builds/web3/
cp index.html builds/

# Clean up
rm -f /tmp/mdbook /tmp/mdbook.tar.gz

echo "Build complete: builds/"
