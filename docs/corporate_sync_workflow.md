# Corporate Sync Workflow

**Internal Documentation - MAGNA Only**

This document describes how to sync student contributions from the external repository to the MAGNA corporate repository.

## Repository Setup

### Remotes

- **`origin`** - Student repository: `git@github-alex-magna:Alex-MAGNA/mi-isal.git`
  - Maintained by students
  - Uses `workflow_dispatch` (manual CI triggers)
  - Students create feature branches and PRs here
  - **PR reviews happen on student's GitHub**

- **`magna-global`** - Corporate repository: `git@github-enterprise:MAGNA-Global/mi-isal.git`
  - Internal MAGNA repository
  - Automatic CI/CD workflows
  - **Only clean, merged states from `origin/main` are synced here**

### Local Branch Tracking

```bash
# Your local main tracks the corporate remote
git branch --show-current  # main
git remote show magna-global  # should show main -> magna-global/main
```

## Workflow Overview

### The Process

1. **Students create feature branches** on their repository (`origin`)
2. **Students open PRs** against `origin/main`
3. **You review PRs externally** on the student's GitHub (outside corporate environment)
4. **Students merge to `origin/main`** after approval
5. **You sync clean state** from `origin/main` to `magna-global/main`

**Key Principle:** Only pull clean, reviewed states that are already merged to `origin/main`.

---

## Syncing Clean States to Corporate

### When to Sync

After students have **merged a PR to their main branch**, you sync that clean state to corporate.

### Sync Process

```bash
# 1. Fetch latest from student repository
git fetch origin

# 2. Review what changed on origin/main
git log main..origin/main --oneline
# Shows all new commits on origin/main that you don't have

# 3. (Optional) View detailed changes before merging
git diff main..origin/main

# 4. Merge student's main into your local main
git checkout main
git merge origin/main
# Note: .gitattributes automatically protects corporate workflows

# 5. Push to corporate repository
git push magna-global main
```

### Verification After Sync

```bash
# Verify sync was successful
git log -3 --oneline

# Confirm corporate and local are in sync
git status
# Should show: "Your branch is up to date with 'magna-global/main'"
```

## Protected Files

The `.gitattributes` file ensures corporate-specific configurations are preserved during merges:

```
.github/workflows/*.yml merge=ours
```

This means:
- ✅ Student workflow changes are **ignored** during merge
- ✅ Corporate CI/CD configuration is **always preserved**
- ✅ No manual conflict resolution needed for workflows

## What NOT to Do

**Never push corporate changes back to student repo:**

```bash
# ❌ DON'T DO THIS
git push origin main  # Would push .gitattributes and corporate workflows to students
```

## Checking Sync Status

```bash
# See if you're ahead/behind student repo
git fetch origin
git log --oneline --graph --decorate main origin/main magna-global/main

# Compare what's different
git diff origin/main main           # Local vs student
git diff main magna-global/main     # Local vs corporate (should be identical)
```
