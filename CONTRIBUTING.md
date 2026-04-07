# Contributing

## Before Opening a PR

- Keep changes focused and minimal.
- Do not hardcode API keys, host addresses, or personal filesystem paths.
- Put generated artifacts under `result/` or `logs/` only.
- Update documentation when script behavior or file layout changes.

## Development Notes

- Install dependencies with `pip install -r requirements.txt`.
- Configure APIs through environment variables documented in `.env.example`.
- Use repository-relative paths so the scripts work on other machines.
