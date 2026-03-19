# Cloud Setup

This file turns the cloud recommendation into an actual deployment path for `research/world_model`.

## Recommendation

Use AWS first:

- region: `us-east-1`
- instance: `g6e.4xlarge`
- AMI: current AWS Deep Learning AMI for Ubuntu 22.04
- storage: `gp3`, 200-500 GB
- access: SSM plus S3
- purchase model: On-Demand

Use Lambda Cloud only if AWS account setup or GPU quota blocks you.

## Official Price Snapshot

Pricing snapshot date: March 9, 2026.

| Provider | Shape | Hourly | 8h/day month | 24/7 month | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| AWS EC2 | `g6e.4xlarge` | `$3.00424` | `~$721.02` | `~$2,163.05` | recommended |
| AWS EC2 | `g5.4xlarge` | `$1.624` | `~$389.76` | `~$1,169.28` | cheaper AWS fallback |
| Lambda Cloud | `1x A6000 48 GB` | `$0.92` | `~$220.80` | `~$662.40` | simpler non-AWS fallback |
| Lambda Cloud | `1x A100 40 GB` | `$1.48` | `~$355.20` | `~$1,065.60` | stronger Lambda fallback |
| Runpod | `RTX 4090` | `from $0.59` | `~$141.60` | `~$424.80` | cheaper, less robust |
| Modal | `L40S` | `~$1.95` | `~$468.00` | `~$1,404.00` | GPU only, CPU and RAM extra |

AWS storage planning:

- gp3 EBS: about `$0.08 / GB-month`
- 200 GB gp3: about `$16 / month`
- 500 GB gp3: about `$40 / month`
- S3 Standard: about `$0.023 / GB-month`
- 100 GB S3 artifacts: about `$2.30 / month`

## AWS Credentials And IAM

Prepare these before using the scripts:

1. Create an S3 bucket for artifacts and repo bundles.
2. Create an EC2 instance role with:
   - `AmazonSSMManagedInstanceCore`
   - S3 read/write access to your artifact bucket
3. Create an operator IAM identity with:
   - `ec2:RunInstances`
   - `ec2:DescribeInstances`
   - `ssm:SendCommand`
   - `ssm:GetCommandInvocation`
   - `iam:PassRole`
   - `s3:PutObject`
   - `s3:GetObject`
   - `s3:ListBucket`
4. Configure local credentials with the AWS CLI or environment variables.

## Installed Tooling

For local cloud automation, install:

```powershell
pip install "research/world_model[cloud]"
```

## Launch Workflow

### 1. Pick an AMI

Use a current AWS Deep Learning AMI in `us-east-1`.

### 2. Launch the EC2 instance

```powershell
$env:PYTHONPATH='e:\Coding\WorldModel\research\world_model\src'
python research/world_model/scripts/aws_launch_instance.py `
  --image-id ami-xxxxxxxxxxxxxxxxx `
  --iam-instance-profile-name GitcgWorldModelEc2Role `
  --security-group-id sg-xxxxxxxxxxxxxxxxx `
  --subnet-id subnet-xxxxxxxxxxxxxxxxx
```

Dry-run the full EC2 request first if needed:

```powershell
$env:PYTHONPATH='e:\Coding\WorldModel\research\world_model\src'
python research/world_model/scripts/aws_launch_instance.py `
  --image-id ami-xxxxxxxxxxxxxxxxx `
  --iam-instance-profile-name GitcgWorldModelEc2Role `
  --security-group-id sg-xxxxxxxxxxxxxxxxx `
  --subnet-id subnet-xxxxxxxxxxxxxxxxx `
  --dry-run
```

### 3. Upload the repo bundle

```powershell
$env:PYTHONPATH='e:\Coding\WorldModel\research\world_model\src'
python research/world_model/scripts/aws_upload_bundle.py `
  --bucket your-artifact-bucket `
  --key bundles/gitcg-world-model-latest.tar.gz
```

### 4. Run the remote training cycle through SSM

```powershell
$env:PYTHONPATH='e:\Coding\WorldModel\research\world_model\src'
python research/world_model/scripts/aws_run_remote_cycle.py `
  --instance-id i-xxxxxxxxxxxxxxxxx `
  --bundle-s3-uri s3://your-artifact-bucket/bundles/gitcg-world-model-latest.tar.gz `
  --artifact-s3-uri s3://your-artifact-bucket/world-model-artifacts/ `
  --wait
```

This remote script does all of the following on the instance:

- downloads the repo bundle from S3
- creates a venv
- installs `research/world_model`
- runs strict simulator integration tests
- runs the iterative self-play cycle
- syncs artifacts back to S3

## Validation Gate

Do not treat the cloud environment as valid until it can:

- launch through API
- pass `GITCG_REQUIRE_INTEGRATION=1`
- generate bootstrap replay
- train a checkpoint
- evaluate a checkpoint
- sync artifacts back to S3

## Lambda Fallback

If AWS is blocked, use Lambda Cloud with:

- `1x A6000 48 GB` first
- `1x A100 40 GB` second
- one persistent instance, not many short jobs
- the same repo bundle and script flow

The current repo is still single-node first, so the AWS scripts are the main path.
