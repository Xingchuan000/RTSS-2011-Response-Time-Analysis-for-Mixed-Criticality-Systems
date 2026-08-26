param(
  [string]$ProjectRoot = ".",
  [string]$ViperRoot,
  [string]$Seeds = "1775,2408,603,814,1555,558,715,2942,313,1012",
  [string]$TreeVariant = "best_overall",
  [string]$ProofRoute = "protected_prefix",
  [int]$MaxParallelSeeds = 4,
  [string]$Python = "python",
  [switch]$Overwrite,
  [switch]$SeedWorker,
  [int]$Seed = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
Set-Location -LiteralPath $ProjectRoot
if ([string]::IsNullOrWhiteSpace($ViperRoot)) {
  throw "-ViperRoot is required and must contain r0_s<seed>\trees\viper\<variant>."
}
$ViperRoot = [System.IO.Path]::GetFullPath($ViperRoot)

function Invoke-Python {
  param([object[]]$Arguments)
  Write-Host "$Python $($Arguments -join ' ')"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed ($LASTEXITCODE): $($Arguments -join ' ')"
  }
}

function New-SeedConfig {
  param([int]$CurrentSeed, [string]$Path)
  $Payload = [ordered]@{
    workload_args = [ordered]@{
      num_tasks = 12
      hi_ratio = 0.5
      period_family = "seed_paired"
      period_scale = 500
      scenario_seed_offset = 100000
      fixed_taskset_seed = $CurrentSeed
      require_schedulable = $true
    }
    runtime_args = [ordered]@{
      runtime_semantics = "C_AMC_SEM"
      end_time = 8000000
      agent_period = 25000
      action_space = "single"
      include_explicit_noop = $true
      budget_increase_ratio = 0.02
      budget_decrease_ratio = 0.02
      budget_floor_ratio = 0.9
      forbid_decreasing_hi_budgets = $true
      mask_detail_mode = "full"
      enable_deploy_cap_mask = $true
      deploy_cap_mask_ratio = 4.0
      deploy_cap_mask_criticality = "lo"
      capture_trace = $true
      capture_debug_events = $false
      processor_overhead = 0
      c_amc_sem_xf = 0.5
    }
    feature_config = [ordered]@{
      observation_mode = "v11_full_10d"
      ema_alpha = 0.2
      overrun_ema_alpha = 0.1
      history_k = 8
      event_window = 10
      max_cost_weight = 0.7
      risk_max_scale = 3.0
      include_safety_margin = $true
    }
    original_reward_mode = "interval_lo_quality_predictive_v4_medium"
    formal_reward_mode = "mendes"
  }
  $Json = ($Payload | ConvertTo-Json -Depth 12) + [Environment]::NewLine
  $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, $Json, $Utf8NoBom)
}

function Invoke-OneSeed {
  param([int]$CurrentSeed)
  $SeedDir = Join-Path $ViperRoot ("r0_s{0}\trees\viper" -f $CurrentSeed)
  $ArtifactDir = Join-Path $SeedDir $TreeVariant
  $Config = Join-Path $SeedDir "formal_seed_config.json"
  $Out = Join-Path $SeedDir ("formal_proof_{0}_{1}" -f $TreeVariant, $ProofRoute)
  foreach ($Name in @("integer_tree.json", "feature_names.json", "action_definitions.json",
                       "fixed_point_config.json", "metadata.json", "artifact_manifest.json")) {
    if (-not (Test-Path -LiteralPath (Join-Path $ArtifactDir $Name))) {
      throw "seed=$CurrentSeed missing artifact file: $Name ($ArtifactDir)"
    }
  }

  New-SeedConfig -CurrentSeed $CurrentSeed -Path $Config
  Invoke-Python @("scripts\bootstrap_formal_target_recipe.py", "--seed-dir", $SeedDir,
    "--seed", $CurrentSeed, "--tree-variant", $TreeVariant, "--config", $Config)
  Invoke-Python @("scripts\regenerate_phase_k_case_map.py", "--out", (Join-Path $SeedDir "phase_k_case_map.json"))
  Invoke-Python @("scripts\export_real_seed_formal_inputs.py", "--seed-dir", $SeedDir,
    "--seed", $CurrentSeed, "--tree-variant", $TreeVariant)

  $ProveArgs = @("-m", "formal_toolchain.cli.prove_seed", "--seed-dir", $SeedDir,
    "--tree-variant", $TreeVariant, "--code-root", $ProjectRoot, "--out", $Out,
    "--proof-route", $ProofRoute, "--refresh-phase-k-map", "--json")
  if ($Overwrite) { $ProveArgs += "--overwrite" }
  Invoke-Python $ProveArgs

  $ResultPath = Join-Path $Out "proof_result.json"
  if (-not (Test-Path -LiteralPath $ResultPath)) { throw "seed=$CurrentSeed proof_result.json missing" }
  $Result = Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json
  if ([string]$Result.result_status -ne "DEPLOYED_TREE_PROVED") {
    throw "seed=$CurrentSeed proof did not succeed: $($Result.result_status)"
  }
  Write-Host "PROVED seed=$CurrentSeed result=$($Result.result_status)" -ForegroundColor Green
}

if ($SeedWorker) {
  if ($Seed -le 0) { throw "-SeedWorker requires -Seed." }
  Invoke-OneSeed -CurrentSeed $Seed
  exit 0
}
if ($MaxParallelSeeds -lt 1) { throw "MaxParallelSeeds must be >= 1." }

$SeedValues = @($Seeds.Split(",") | ForEach-Object { [int]$_.Trim() })
$Pending = [System.Collections.Generic.Queue[int]]::new()
foreach ($Value in $SeedValues) { $Pending.Enqueue($Value) }
$Running = @{}
$Failures = @()

while ($Pending.Count -gt 0 -or $Running.Count -gt 0) {
  while ($Pending.Count -gt 0 -and $Running.Count -lt $MaxParallelSeeds) {
    $Current = $Pending.Dequeue()
    $Job = Start-Job -ScriptBlock {
      param($Script, $Root, $Viper, $Variant, $Route, $Py, $CurrentSeed, $DoOverwrite)
      $Arguments = @("-ProjectRoot", $Root, "-ViperRoot", $Viper, "-TreeVariant", $Variant,
        "-ProofRoute", $Route, "-Python", $Py, "-SeedWorker", "-Seed", $CurrentSeed)
      if ($DoOverwrite) { $Arguments += "-Overwrite" }
      & $Script @Arguments
    } -ArgumentList $PSCommandPath, $ProjectRoot, $ViperRoot, $TreeVariant,
      $ProofRoute, $Python, $Current, ([bool]$Overwrite)
    $Running[$Job.Id] = [pscustomobject]@{ Job = $Job; Seed = $Current }
    Write-Host "START seed=$Current running=$($Running.Count)/$MaxParallelSeeds"
  }

  $Done = Wait-Job -Job @($Running.Values.Job) -Any
  $Entry = $Running[$Done.Id]
  Receive-Job -Job $Done
  if ($Done.State -ne "Completed") { $Failures += $Entry.Seed }
  Remove-Job -Job $Done -Force
  $Running.Remove($Done.Id)
}

if ($Failures.Count -gt 0) {
  throw "Formal proof failed for seeds: $($Failures -join ',')"
}
Write-Host "All formal proofs completed. seeds=$($SeedValues -join ',')" -ForegroundColor Green
