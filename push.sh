#!/bin/bash
git add .

git status

COMMIT_MSG=${1:-"update"}
git commit -m "$COMMIT_MSG"

git push origin main