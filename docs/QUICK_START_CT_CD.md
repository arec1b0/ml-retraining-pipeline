# Quick Start: CT-CD Integration

## Overview

This guide helps you quickly set up the automated Continuous Training → Continuous Deployment pipeline integration.

## 5-Minute Setup

### Step 1: Create GitHub Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name like "ML Pipeline CD Trigger"
4. Select scopes:
   - ✅ `repo` (Full control of private repositories)
   - ✅ `workflow` (Update GitHub Action workflows)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)

### Step 2: Configure Environment Variables

Create or update your `.env` file in the project root:

```bash
# Enable CT-CD Integration
ENABLE_CD_TRIGGER=true

# GitHub Configuration
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GITHUB_REPO_OWNER=your-username
GITHUB_REPO_NAME=ml-retraining-pipeline

# (Optional) Custom workflow name
CD_WORKFLOW_NAME=cd_pipeline.yml
```

### Step 3: Test the Integration

```bash
# Run the retraining pipeline
prefect deployment run 'Model Retraining Flow/production-retraining'
```

### Step 4: Verify in GitHub Actions

1. Go to your GitHub repository
2. Click on the "Actions" tab
3. Look for a workflow run with "automated_ct_pipeline" trigger

## What Happens Automatically?

```
┌─────────────────────────────────────────┐
│  1. New data arrives                    │
│  2. CT pipeline trains new model        │
│  3. New model outperforms current       │
│  4. Model promoted to Production        │
│  5. 🚀 CD pipeline triggered           │
│  6. Docker image built & pushed         │
│  7. Ready for deployment!               │
└─────────────────────────────────────────┘
```

## Disable Auto-Deployment

Set in `.env`:
```bash
ENABLE_CD_TRIGGER=false
```

## Troubleshooting

### "CD pipeline trigger is disabled"
→ Set `ENABLE_CD_TRIGGER=true` in `.env`

### "GitHub configuration incomplete"
→ Ensure all three variables are set: `GITHUB_TOKEN`, `GITHUB_REPO_OWNER`, `GITHUB_REPO_NAME`

### "Failed to trigger CD pipeline. Status code: 401"
→ GitHub token is invalid or expired. Generate a new one.

### "Failed to trigger CD pipeline. Status code: 404"
→ Check repository owner/name spelling. Ensure token has access to the repo.

## Full Documentation

For detailed architecture, security best practices, and advanced configuration:

📖 **[Full CT-CD Integration Documentation](./CT_CD_INTEGRATION.md)**

## Security Note

⚠️ **NEVER commit your `.env` file or GitHub token to version control!**

The `.env` file is already in `.gitignore` to protect your secrets.

