#!/usr/bin/env bash
# Bump the package version, push, and trigger JuliaRegistrator with
# AI-generated release notes.
#
# Usage: scripts/release.sh <patch|minor|major|X.Y.Z> [options]
#
# After this script finishes:
#   1. JuliaRegistrator opens a PR against JuliaRegistries/General, with the
#      generated release notes attached.
#   2. Once that PR merges (usually automatic within ~15-30 min), TagBot
#      creates the git tag and GitHub release automatically, carrying the
#      same release notes over.
# Do NOT create the tag manually in the meantime - that pre-empts TagBot
# and leaves the release un-created (this happened for v0.8.2).

set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=true
AI_TOOL="claude"
NO_AI=false
BUMP=""

usage() {
    cat <<EOF
Usage: $(basename "$0") <patch|minor|major|X.Y.Z> [OPTIONS]

Options:
  --ai {claude,codex,opencode}  AI tool for release notes (default: claude)
  --no-ai                       Skip AI notes, use a plain commit-list instead
  --execute                     Actually run (default is dry-run)
  -h, --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ai) AI_TOOL="$2"; shift 2 ;;
        --no-ai) NO_AI=true; shift ;;
        --execute) DRY_RUN=false; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *)
            if [[ -n "$BUMP" ]]; then
                echo "Unexpected extra argument: $1" >&2
                exit 1
            fi
            BUMP="$1"
            shift
            ;;
    esac
done

if [[ -z "$BUMP" ]]; then
    usage
    exit 1
fi

if [[ ! "$AI_TOOL" =~ ^(claude|codex|opencode)$ ]]; then
    echo "Error: --ai must be claude, codex, or opencode (got '$AI_TOOL')" >&2
    exit 1
fi

# --- Guard: clean working tree on main ---
branch=$(git branch --show-current)
if [[ "$branch" != "main" ]]; then
    echo "Must be on main branch (currently on $branch)" >&2
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Working tree has uncommitted changes to tracked files, aborting" >&2
    exit 1
fi

git pull --ff-only

# --- Bump version ---
current=$(grep -m1 '^version' Project.toml | sed -E 's/version = "(.*)"/\1/')
IFS='.' read -r major minor patch <<< "$current"

case "$BUMP" in
    patch) new="$major.$minor.$((patch + 1))" ;;
    minor) new="$major.$((minor + 1)).0" ;;
    major) new="$((major + 1)).0.0" ;;
    [0-9]*.[0-9]*.[0-9]*) new="$BUMP" ;;
    *)
        echo "Invalid argument: $BUMP (expected patch|minor|major|X.Y.Z)" >&2
        exit 1
        ;;
esac
tag="v${new}"

echo "Version bumped to $new (tag: $tag)"

run() {
    echo "+ $*"
    if [[ "$DRY_RUN" == true ]]; then
        return
    fi
    "$@"
}

# --- Generate release notes ---
PREV_TAG=$(git describe --tags --abbrev=0 HEAD 2>/dev/null || echo "")
if [[ -n "$PREV_TAG" ]]; then
    COMMITS=$(git log "${PREV_TAG}..HEAD" --pretty=format:'- %s%n%b' -- . ':!Project.toml')
else
    COMMITS=$(git log --pretty=format:'- %s%n%b' -20 -- . ':!Project.toml')
fi

NOTES=""

if [[ "$NO_AI" == false ]]; then
    PROMPT="You are writing release notes for the Julia package BlochSimulators, version $tag (a package for Bloch simulations in the context of Magnetic Resonance Imaging).

Here are the commits since the last release ($PREV_TAG):

$COMMITS

Write concise, user-facing release notes in this exact format:

## Changes

- **Short title**: one or two sentence description

Rules:
- Write for downstream users of the package (MRI researchers), not for the package's own contributors - no commit hashes, no file names, no internal function names unless part of the public API
- 3-8 bullet points max - group related commits into one bullet
- Use past tense (\"fixed\", \"added\", \"changed\")
- Skip pure refactors, docs, or CI changes unless they affect users
- If any change alters the numerical output of existing functionality (e.g. a sign-convention or physics-correctness fix), put it first, prefix the bullet with '**BREAKING**', and explain what a user needs to check or update in their own code
- Bold the short title, keep the description tight"

    case "$AI_TOOL" in
        claude)
            if command -v claude &>/dev/null; then
                echo "Generating release notes with Claude..."
                NOTES=$(claude -p "$PROMPT" 2>/dev/null) || true
            fi
            ;;
        codex)
            if command -v codex &>/dev/null; then
                echo "Generating release notes with Codex..."
                tmpfile=$(mktemp)
                errfile=$(mktemp)
                trap 'rm -f "$tmpfile" "$errfile"' EXIT
                if printf '%s\n' "$PROMPT" | codex exec --output-last-message "$tmpfile" - >/dev/null 2>"$errfile"; then
                    NOTES=$(<"$tmpfile")
                elif [[ -s "$errfile" ]]; then
                    echo "Codex release notes failed:"
                    sed 's/^/  /' "$errfile"
                fi
            fi
            ;;
        opencode)
            if command -v opencode &>/dev/null; then
                echo "Generating release notes with opencode..."
                NOTES=$(printf '%s\n' "$PROMPT" | opencode run - 2>/dev/null) || true
            fi
            ;;
    esac
fi

if [[ -z "$NOTES" ]]; then
    if [[ "$NO_AI" == false ]]; then
        echo "AI notes unavailable, falling back to commit-based notes"
    fi
    NOTES="## Changes"
    if [[ -n "$COMMITS" ]]; then
        NOTES+=$'\n'"$COMMITS"
    else
        NOTES+=$'\n'"- Maintenance release."
    fi
fi

echo ""
echo "--- Release notes preview ---"
echo "$NOTES"
echo "------------------------------"
echo ""

# --- Commit, push, request registration ---
sed -i -E "s/^version = \".*\"/version = \"$new\"/" Project.toml

run git add Project.toml
run git commit -m "Release $tag"
run git push origin main

if [[ "$DRY_RUN" == true ]]; then
    echo "(dry-run - re-run with --execute to apply)"
    git checkout Project.toml
    exit 0
fi

sha=$(git rev-parse HEAD)
repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)

comment_body="@JuliaRegistrator register

Release notes:

$NOTES"

echo "Requesting registration for $tag on commit $sha ($repo)..."
gh api "repos/$repo/commits/$sha/comments" -f body="$comment_body" >/dev/null

echo ""
echo "Done. JuliaRegistrator will comment back with a link to the JuliaRegistries/General PR."
echo "Once merged, TagBot creates the tag + GitHub release automatically, with these notes attached."
