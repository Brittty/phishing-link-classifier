# =============================================================================
# push-to-github.ps1
# Initializes a git repo, commits all project files, and prepares for push to GitHub.
#
# Prerequisites:
#   1. Git installed (https://git-scm.com/download/win)
#   2. A GitHub repository created at https://github.com/new
#      - Name: phishing-link-classifier
#      - Do NOT initialize with README, .gitignore, or license
#   3. GitHub CLI (gh) authenticated, OR a Personal Access Token (PAT)
#
# Usage:
#   .\push-to-github.ps1 -GitHubUsername "your-username"
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GitHubUsername,

    [Parameter(Mandatory = $false)]
    [string]$RepositoryName = "phishing-link-classifier",

    [Parameter(Mandatory = $false)]
    [string]$Branch = "main",

    [Parameter(Mandatory = $false)]
    [switch]$UseGitHubCLI
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Get-Location
$RepoUrl = "https://github.com/$GitHubUsername/$RepositoryName.git"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Phishing Link Classifier - GitHub Push Helper" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check git is installed
Write-Host "[1/6] Checking git installation..." -ForegroundColor Yellow
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git is not installed. Please install from https://git-scm.com/download/win and re-run this script."
}
git --version
Write-Host ""

# Step 2: Check for uncommitted changes / existing repo
Write-Host "[2/6] Checking repository state..." -ForegroundColor Yellow
if (Test-Path ".git") {
    Write-Host "  Git repo already initialized." -ForegroundColor Green
} else {
    Write-Host "  Initializing new git repository..." -ForegroundColor Green
    git init
}
Write-Host ""

# Step 3: Configure user if not set
Write-Host "[3/6] Checking git user config..." -ForegroundColor Yellow
$userName = git config user.name
$userEmail = git config user.email
if (-not $userName) {
    git config user.name "$GitHubUsername"
    Write-Host "  Set user.name to $GitHubUsername" -ForegroundColor Green
}
if (-not $userEmail) {
    git config user.email "$GitHubUsername@users.noreply.github.com"
    Write-Host "  Set user.email to $GitHubUsername@users.noreply.github.com" -ForegroundColor Green
}
Write-Host ""

# Step 4: Add and commit
Write-Host "[4/6] Staging and committing files..." -ForegroundColor Yellow
git add .
$status = git status --porcelain
if (-not $status) {
    Write-Host "  No changes to commit." -ForegroundColor Yellow
} else {
    git commit -m "Initial commit: offline phishing link classifier with scikit-learn

- Modular src/ layout: config, security, features, dataset, model, evaluate, predict, cli
- Deterministic offline URL feature extraction (29 features)
- Logistic Regression vs Random Forest comparison with GridSearchCV
- Stratified + grouped-by-domain train/val/test splits with leakage prevention
- Validation-based threshold selection; isolated test set evaluation
- CLI: single URL, batch CSV, JSON/CSV output
- Security: input validation, query param redaction, trusted artifact loading
- 37 unit tests covering features, dataset, model, CLI, and batch processing
- Dockerfile, .dockerignore, GitHub Actions CI, PyPI publish workflow
- pyproject.toml (PEP 621) packaging with console script entry point"
    Write-Host "  Commit created successfully." -ForegroundColor Green
}
Write-Host ""

# Step 5: Set branch and remote
Write-Host "[5/6] Configuring branch and remote..." -ForegroundColor Yellow
git branch -M $Branch

# Check if remote already exists
$remoteExists = git remote | Where-Object { $_ -eq "origin" }
if ($remoteExists) {
    Write-Host "  Remote 'origin' already exists. Updating URL..." -ForegroundColor Yellow
    git remote set-url origin $RepoUrl
} else {
    git remote add origin $RepoUrl
    Write-Host "  Remote 'origin' added: $RepoUrl" -ForegroundColor Green
}
Write-Host ""

# Step 6: Push
Write-Host "[6/6] Pushing to GitHub..." -ForegroundColor Yellow
if ($UseGitHubCLI) {
    Write-Host "  Using GitHub CLI (gh) for authentication..." -ForegroundColor Green
    gh auth login --hostname github.com --git-protocol https 2>$null
    git push -u origin $Branch
} else {
    Write-Host "  Pushing via HTTPS. You may be prompted for your GitHub username and PAT." -ForegroundColor Yellow
    Write-Host "  Note: Use a Personal Access Token (PAT) as the password, not your GitHub password." -ForegroundColor Yellow
    git push -u origin $Branch
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  SUCCESS! Your project is now on GitHub:" -ForegroundColor Green
Write-Host "  $RepoUrl" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Visit https://github.com/$GitHubUsername/$RepositoryName to verify" -ForegroundColor Cyan
Write-Host "  2. Enable GitHub Actions in the Actions tab" -ForegroundColor Cyan
Write-Host "  3. Configure PyPI publishing secrets if publishing to PyPI" -ForegroundColor Cyan