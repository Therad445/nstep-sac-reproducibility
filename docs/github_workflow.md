# GitHub Workflow

The project uses larger, logical commits instead of tiny file-by-file commits.
Commit messages follow Conventional Commits.

## First repository setup

```bash
git init
git branch -M main
git add -A
git commit -m "chore: scaffold n-step sac reproducibility project"
```

Create and push the public GitHub repository:

```bash
gh auth status
gh repo create Therad445/nstep-sac-reproducibility \
  --public \
  --source=. \
  --remote=origin \
  --push
```

If the repository already exists:

```bash
git remote add origin git@github.com:Therad445/nstep-sac-reproducibility.git
git push -u origin main
```

## Planned commit blocks

```text
chore: scaffold n-step sac reproducibility project
feat: add sac baseline training and logging
feat: add n-step return ablation configs
exp: add pendulum ablation results
docs: add paper notes and weakness analysis
exp: add halfcheetah ablation results
docs: prepare final presentation outline and reproduction card
```

`exp` is used as a project-local type for experiment logs, tables and plots.

## What should not be committed

Do not commit virtual environments, model checkpoints, `.env` files, raw large runs, or local caches.
Curated small plots and summary tables can be committed because they are part of the final report.
