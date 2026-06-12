# Deploy runbook (step 6) — taking the harvester live on AWS

The operator checklist for a first real deployment. Everything in steps 1–5
(code migration, SageMaker handler, Docker image, Terraform, CI) is done and
validated **offline**; this runbook covers what can only be done against a live
AWS account, in order, with the ordering gotchas called out.

> Prereqs on the operator's machine: AWS CLI with **admin** creds for the target
> account (the first apply needs them — see §0), Terraform ≥ 1.6, Docker, `uv`,
> and the fine-tuned ranker checkpoint (`RANKER_PATH` with `adapter/`,
> `tokenizer/`, `score_head.pt`).

All `terraform` commands run from `infra/terraform/`. Resource/var/output/secret
names below are the real ones from the stack.

---

## 0. Bootstrap the things Terraform can't create for itself

1. **Pick a region** where **Bedrock Nova 2 Lite is available** and **request
   model access** for it in the Bedrock console (access is not on by default).
   Use this region everywhere below as `$REGION`.

2. **Remote state** — create an S3 bucket + DynamoDB lock table once (names are
   yours; they feed `-backend-config`):
   ```bash
   aws s3api create-bucket --bucket <tf-state-bucket> --region $REGION
   aws s3api put-bucket-versioning --bucket <tf-state-bucket> \
     --versioning-configuration Status=Enabled
   aws dynamodb create-table --table-name <tf-lock-table> \
     --attribute-definitions AttributeName=LockID,AttributeType=S \
     --key-schema AttributeName=LockID,KeyType=HASH \
     --billing-mode PAY_PER_REQUEST --region $REGION
   ```

3. **Pick the SageMaker DLC image** (`ranker_image_uri`) — a **PyTorch GPU
   inference DLC** in `$REGION` whose torch matches `uv.lock` (2.11.x). Get the
   URI from the AWS Deep Learning Containers list for your region.

---

## 1. The ordering gotcha (read before applying)

Two resources have a **bootstrap circularity**:

- The **ranker S3 bucket** is *created by* Terraform, but `ranker_model_s3_uri`
  (the artifact in that bucket) is a *required input* to the same stack, and the
  **SageMaker model/endpoint** can't come up without the artifact present.

So apply in **two phases**: first create the bucket + ECR, then upload the
artifact + push the image, then apply the rest.

---

## 2. Phase-1 apply — create the registries (bucket + ECR + state plumbing)

```bash
terraform init \
  -backend-config="bucket=<tf-state-bucket>" \
  -backend-config="dynamodb_table=<tf-lock-table>" \
  -backend-config="region=$REGION"

# Create just the ranker bucket and the ECR repo first. Placeholder values for the
# two required vars are fine here — the targeted resources don't read them.
terraform apply \
  -var "region=$REGION" \
  -var "budget_alert_email=<you@example.com>" \
  -var "ranker_model_s3_uri=s3://placeholder/model.tar.gz" \
  -var "ranker_image_uri=placeholder" \
  -target=aws_s3_bucket.ranker \
  -target=aws_ecr_repository.harvester

RANKER_BUCKET=$(terraform output -raw ranker_bucket)
ECR_URL=$(terraform output -raw ecr_repository_url)
```

## 3. Build + upload the two artifacts

**Ranker model** → the bucket from phase 1:
```bash
../sagemaker/package_model.sh "$RANKER_PATH" "s3://$RANKER_BUCKET/model/model.tar.gz"
```

**Harvester image** → ECR (locally for the first deploy; CI does this later).
Fetch/stage `.chroma/` so `COPY .chroma/` succeeds (it's gitignored):
```bash
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin "${ECR_URL%/*}"
docker build -t "$ECR_URL:bootstrap" ..        # repo root has the Dockerfile
docker push "$ECR_URL:bootstrap"
```
(Optionally publish `.chroma/` to S3 and set the `CHROMA_S3_URI` repo var so CI
can rebuild the image later.)

## 4. Phase-2 apply — the full stack

```bash
terraform apply \
  -var "region=$REGION" \
  -var "budget_alert_email=<you@example.com>" \
  -var "ranker_model_s3_uri=s3://$RANKER_BUCKET/model/model.tar.gz" \
  -var "ranker_image_uri=<the DLC image uri from §0.3>" \
  -var "image_tag=bootstrap"
```

This stands up the VPC + endpoints, the SageMaker async endpoint, DynamoDB, the
task definition, the EventBridge schedule, secrets (empty), and the budget alarm.
Expect the SageMaker endpoint to take several minutes to reach `InService`.

## 5. Populate secrets (never in Terraform)

```bash
PFX=cmv-harvester
aws secretsmanager put-secret-value --secret-id $PFX/tavily-api-key --secret-string '<tavily key>'
aws secretsmanager put-secret-value --secret-id $PFX/openai-api-key  --secret-string '<openai key>'
# ntfy is a JSON secret (the task def reads :topic:: and :token:: from it):
aws secretsmanager put-secret-value --secret-id $PFX/ntfy \
  --secret-string '{"topic":"cmv-<random>","token":"tk_<...>"}'
```
Confirm the budget SNS subscription email and the ntfy topic subscription.

---

## 6. Smoke-test, in increasing blast radius

**a. Ranker endpoint** — score one pair directly:
```bash
echo '{"topic":"t","post":"p","arg_a":"a strong argument","arg_b":"a weak one"}' \
  > /tmp/req.json
aws s3 cp /tmp/req.json "s3://$RANKER_BUCKET/ranker-requests/smoke.json"
aws sagemaker-runtime invoke-endpoint-async \
  --endpoint-name "$(terraform output -raw sagemaker_endpoint_name)" \
  --input-location "s3://$RANKER_BUCKET/ranker-requests/smoke.json" \
  --content-type application/json --region $REGION
# then read the result object under s3://$RANKER_BUCKET/async/output/
```
Expect `{score_a, score_b, winner, prob_a_better}`. **This is also the first real
check that Bedrock Nova 2 Lite isn't involved yet — it isolates the ranker.**

**b. One harvester run, dry (no spend, no ledger writes)** — run the task with a
`--dry-run` command override:
```bash
aws ecs run-task --cluster cmv-harvester-cluster \
  --task-definition cmv-harvester-harvester --launch-type FARGATE \
  --network-configuration '{"awsvpcConfiguration":{"subnets":[<private subnet ids>],"securityGroups":[<task sg>],"assignPublicIp":"DISABLED"}}' \
  --overrides '{"containerOverrides":[{"name":"harvester","command":["python","-m","harvester.orchestrate","--dry-run"]}]}' \
  --region $REGION
```
Tail CloudWatch (`terraform output -raw log_group`). A dry run searches +
classifies only — this exercises Bedrock (the classifier) and the Fediverse
egress without drafting or notifying.

**c. One real run, capped** — same as (b) without `--dry-run` (defaults already
cap to `--max-generations 3`, `--max-age-hours 24`). Verify: a CloudWatch log
with post ids only (no secrets), a row in the `responses` table, an ntfy push.

**d. Let the schedule fire** once on its own (`rate(1 hour)`), then confirm the
dedup ledger prevents re-answering on the next tick.

---

## 7. Validate the things only a live run can show

- [ ] **Bedrock Nova 2 Lite output quality.** The graph's prompts were written for
      gpt-5.4-nano; behaviour *will* shift. Read a few generated rebuttals — are
      they coherent, pro-Israel, grounded? This is the single biggest unknown the
      offline suite could not cover. Re-tune prompts in `agents/graph/chains/` if
      the stance/grounding gates trip too often or quality drops.
- [ ] **Ranker parity.** Spot-check that the SageMaker endpoint's winner matches
      the local ranker on a couple of pairs (run the local path with `RANKER_PATH`
      set and compare) — confirms the handler reproduces `score_pair`.
- [ ] **Least-privilege holds.** The run succeeds with the scoped task role (no
      AccessDenied in logs); if something's denied, fix the *policy*, don't widen
      to a wildcard.
- [ ] **Cost.** After a day, check the Budgets actual vs. the SageMaker
      scale-to-zero behaviour (the endpoint should drop to 0 instances between
      runs).

## 8. Hand CI the wheel

Once a manual deploy works:
1. Run the **first** `terraform apply` locally (done above) — it created the
   `cmv-harvester-ci` OIDC role.
2. Set the GitHub repo **variables** (Settings → Secrets and variables → Actions)
   listed in `terraform/README.md` (`AWS_ROLE_ARN` = `ci_role_arn` output, region,
   state bucket/table, `ECR_REPOSITORY`, the ranker URIs, budget email, optional
   `CHROMA_S3_URI`).
3. Add **branch protection** on `main` requiring the `tests` check, so a red suite
   blocks merges (the workflows are otherwise independent).
4. From here: PRs run `tests` + terraform `plan`; merges to `main` `apply` and
   push a new image tagged with the commit sha.

---

## Rollback / teardown

- **Bad image:** set `image_tag` back to a known-good sha and re-apply (or pin it
  in the task def); the next scheduled run uses it.
- **Stop all spend fast:** disable the schedule
  (`aws scheduler update-schedule ... --state DISABLED`) and set the ranker
  endpoint's autoscaling min to 0 (already the floor) / delete the endpoint.
- **Full teardown:** `terraform destroy` (empty the ranker S3 bucket first — S3
  buckets won't delete while non-empty). The remote-state bucket/table from §0
  are not managed by this stack; remove them by hand if desired.
