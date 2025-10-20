# Continuous Training → Continuous Deployment Integration

## Overview

This document describes the automated CT (Continuous Training) to CD (Continuous Deployment) pipeline linkage that enables fully automated model retraining and deployment workflows.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CT Pipeline (Prefect)                        │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Ingest  │→ │  Train   │→ │ Evaluate │→ │   Register   │  │
│  │   Data   │  │  Model   │  │  Model   │  │   & Promote  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
│                                                    │             │
│                                                    ↓             │
│                                          ┌──────────────────┐   │
│                                          │ Model Promoted   │   │
│                                          │ to Production?   │   │
│                                          └──────────────────┘   │
│                                                    │             │
└────────────────────────────────────────────────────│─────────────┘
                                                     │
                                                     ↓
                                          ┌──────────────────┐
                                          │  GitHub Actions  │
                                          │  workflow_dispatch│
                                          │      API Call    │
                                          └──────────────────┘
                                                     │
┌────────────────────────────────────────────────────│─────────────┐
│                    CD Pipeline (GitHub Actions)    ↓             │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Build   │→ │   Test   │→ │   Push   │→ │    Deploy    │  │
│  │  Docker  │  │  Image   │  │    to    │  │  Inference   │  │
│  │  Image   │  │          │  │  Registry│  │   Service    │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Model Training & Promotion

When the CT pipeline runs:

1. A new model is trained on fresh data
2. The model is evaluated against validation metrics
3. If the new model outperforms the current Production model:
   - The new model is promoted to "Production" stage in MLflow
   - The old model is archived
   - **The `trigger_cd_pipeline()` function is automatically called**

### 2. CD Pipeline Trigger

The `trigger_cd_pipeline()` function:

- Makes an authenticated API call to GitHub Actions
- Uses the `workflow_dispatch` event to trigger the CD pipeline
- Passes metadata about the promoted model (version, accuracy)
- Logs the trigger status in Prefect

### 3. Automated Deployment

The CD pipeline then:

1. Builds a new Docker image for the inference service
2. Tags it with the commit SHA and `latest`
3. Pushes it to GitHub Container Registry
4. Makes it available for deployment

## Setup Instructions

### Prerequisites

1. **GitHub Personal Access Token** with the following scopes:
   - `repo` (Full control of private repositories)
   - `workflow` (Update GitHub Action workflows)

   Create one at: https://github.com/settings/tokens

2. **GitHub Repository** with:
   - The CD workflow file (`.github/workflows/cd_pipeline.yml`)
   - Repository secrets configured (if deploying to production)

### Configuration Steps

#### 1. Create/Update Your `.env` File

Add the following configuration to your `.env` file:

```bash
# Enable CT -> CD integration
ENABLE_CD_TRIGGER=true

# GitHub Configuration
GITHUB_TOKEN=ghp_your_personal_access_token_here
GITHUB_REPO_OWNER=your-github-username
GITHUB_REPO_NAME=ml-retraining-pipeline

# Optional: Custom workflow name (if you renamed it)
CD_WORKFLOW_NAME=cd_pipeline.yml
```

#### 2. Update CD Workflow Permissions (if needed)

Ensure your CD workflow has necessary permissions in `.github/workflows/cd_pipeline.yml`:

```yaml
jobs:
  build-and-push-inference-service:
    name: Build and Push Inference Service
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
```

#### 3. Test the Integration

You can test the integration by:

**Option A: Run the CT Pipeline**
```bash
# Run the full retraining pipeline
prefect deployment run 'Model Retraining Flow/production-retraining'
```

**Option B: Manual Trigger (Testing)**
```bash
# Use the GitHub CLI to manually trigger
gh workflow run cd_pipeline.yml \
  -f model_version="5" \
  -f model_accuracy="0.9234" \
  -f trigger_source="manual_test"
```

**Option C: Via GitHub UI**
1. Go to your repository on GitHub
2. Navigate to `Actions` → `CD Pipeline`
3. Click `Run workflow`
4. Fill in the optional parameters
5. Click `Run workflow`

## Security Best Practices

### 1. Token Security

- **Never commit** your GitHub token to version control
- Use environment variables or secrets management
- Rotate tokens regularly (every 90 days)
- Use fine-grained personal access tokens with minimal scopes

### 2. Validation & Error Handling

The implementation includes:

- ✅ Configuration validation before making API calls
- ✅ Graceful degradation if CD trigger fails
- ✅ Detailed logging for troubleshooting
- ✅ Network timeout protection (30 seconds)
- ✅ Proper error messages for debugging

### 3. Production Considerations

For production deployments, consider:

1. **Use GitHub Apps** instead of personal access tokens
2. **Implement approval gates** in your CD pipeline
3. **Add deployment notifications** (Slack, email)
4. **Monitor deployment success/failure**
5. **Implement rollback mechanisms**

## Monitoring & Troubleshooting

### Checking Integration Status

**In Prefect Logs:**
```
🔗 Initiating CT -> CD pipeline linkage...
✅ Successfully triggered CD pipeline! Check GitHub Actions for deployment status.
```

**In GitHub Actions:**
1. Go to `Actions` tab in your repository
2. Look for workflow runs with trigger source: `automated_ct_pipeline`
3. Check the deployment summary for model metadata

### Common Issues

#### Issue: "CD pipeline trigger is disabled"

**Solution:** Set `ENABLE_CD_TRIGGER=true` in your `.env` file

#### Issue: "GitHub configuration incomplete"

**Solution:** Ensure all required GitHub settings are configured:
```bash
GITHUB_TOKEN=ghp_...
GITHUB_REPO_OWNER=your-username
GITHUB_REPO_NAME=repo-name
```

#### Issue: "Failed to trigger CD pipeline. Status code: 401"

**Solution:** Your GitHub token is invalid or expired. Generate a new one with correct scopes.

#### Issue: "Failed to trigger CD pipeline. Status code: 404"

**Solution:** Check that:
- Repository owner and name are correct
- Workflow file name matches `CD_WORKFLOW_NAME`
- Token has access to the repository

#### Issue: "Network error while triggering CD pipeline"

**Solution:** Check your internet connection and GitHub API status at https://www.githubstatus.com/

## Disabling the Integration

To disable automatic CD triggering:

1. Set in your `.env`:
   ```bash
   ENABLE_CD_TRIGGER=false
   ```

2. Models will still be promoted to Production in MLflow, but the CD pipeline won't be triggered automatically

3. You can manually trigger deployments via GitHub Actions UI or CLI

## Advanced Configuration

### Custom Workflow Inputs

You can extend the workflow_dispatch inputs in `.github/workflows/cd_pipeline.yml`:

```yaml
workflow_dispatch:
  inputs:
    model_version:
      description: "Model version promoted to Production"
      required: false
      type: string
    model_accuracy:
      description: "Model accuracy"
      required: false
      type: string
    trigger_source:
      description: "Source that triggered the deployment"
      required: false
      type: string
      default: "manual"
    # Add custom inputs here
    deployment_environment:
      description: "Target deployment environment"
      required: false
      type: choice
      options:
        - staging
        - production
      default: "production"
```

### Conditional Deployment

Modify `src/pipeline/tasks/register.py` to add conditions:

```python
# Only trigger CD for production-ready models
if new_accuracy > 0.90 and settings.ENABLE_CD_TRIGGER:
    cd_triggered = trigger_cd_pipeline(...)
```

## Related Documentation

- [CD Pipeline README](../inference-service/README.md) - Inference service deployment details
- [Prefect Deployment Guide](../README.md) - CT pipeline setup
- [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html) - Model lifecycle management

## Architecture Benefits

This CT-CD integration provides:

1. **🚀 Zero-Touch Deployment** - Models are automatically deployed after successful training
2. **🔒 Validation Gates** - Only better models are promoted and deployed
3. **📊 Full Traceability** - Complete audit trail from training to deployment
4. **⚡ Fast Iteration** - Reduce time from model training to production
5. **🛡️ Safety** - Graceful degradation if deployment fails
6. **📝 Visibility** - Comprehensive logging at every step

## Future Enhancements

Potential improvements to the integration:

- [ ] Add deployment status webhook back to Prefect
- [ ] Implement blue-green deployment strategy
- [ ] Add automatic rollback on deployment failure
- [ ] Support multiple deployment targets (staging, production)
- [ ] Add deployment approval workflow for critical environments
- [ ] Integrate with Kubernetes for dynamic scaling
- [ ] Add canary deployment support
- [ ] Implement deployment metrics tracking

