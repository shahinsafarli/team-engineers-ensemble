## Summary

Briefly explain what this PR changes.

## Files changed

- `path/to/file.py`
- `path/to/test_file.py`

## Testing done

Paste commands you ran:

```bash
python -m pytest -q
python -m pytest --cov=src --cov-report=term-missing --cov-fail-under=60 -q
python -m mypy src --ignore-missing-imports
python -m ruff check src tests
```

Before final CI activation, attach or summarize the successful local gate output. Once GitHub Actions is added in the final documentation PR, confirm that its quality-gate job is green.

## Checklist

- [ ] I worked on a feature/fix/docs branch, not directly on `main`.
- [ ] I pulled latest `main` before opening this PR.
- [ ] I changed only files related to my task.
- [ ] I ran relevant tests locally.
- [ ] I did not commit cache files, `.pyc`, `.coverage`, `.zip`, or local junk.
- [ ] I did not break the official project structure.
- [ ] I did not use forbidden sklearn implementations as project implementations.
- [ ] Study metadata contains only the four required datasets with exact sizes 569, 48,842, 50,000, and 14,780.
- [ ] Covertype has seven targets with the locked seed-42 allocation; no study path reduces required data.
- [ ] No empirical output uses synthetic data, Digits, or binary rare-class Covertype.
- [ ] Fitted preprocessing and oversampling use training rows only for supervised evaluation.
- [ ] Generated artifacts were produced by the runner and their provenance/row counts were reviewed.
- [ ] I can explain and defend the code/content in this PR.

## Reviewer notes

Mention anything the reviewer should check carefully.
