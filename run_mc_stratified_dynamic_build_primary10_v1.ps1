param(
  [string]$ProjectRoot = ".",
  [string]$Python = "python",
  [int]$CandidateCount = 3000,
  [switch]$RequireSchedulable,
  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$OutRoot = Join-Path $ProjectRoot "outputs\tasksets\mc_stratified_dynamic_v1"
$Manifest = Join-Path $OutRoot "candidates.csv"
$Rejections = Join-Path $OutRoot "candidate_rejections.csv"
$D0 = Join-Path $OutRoot "diagnostics_d0.csv"
$D1 = Join-Path $OutRoot "diagnostics_d1.csv"
$D2 = Join-Path $OutRoot "diagnostics_d2.csv"
$Primary10 = Join-Path $OutRoot "mc_stratified_dynamic_primary10_v1.csv"
$Audit = Join-Path $OutRoot "primary10_selection_audit.csv"
$Shortage = Join-Path $OutRoot "primary10_shortage_report.json"

function Invoke-PythonStep {
  param([string]$Label, [string[]]$Arguments)
  Write-Host "${Label}: $Python $($Arguments -join ' ')"
  if ($DryRun) { return }
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "${Label}: python command failed with exit code $LASTEXITCODE"
  }
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$SchedArgs = @()
if ($RequireSchedulable) { $SchedArgs += "--require-schedulable" }

Invoke-PythonStep -Label "generate candidates 0..$($CandidateCount - 1)" -Arguments (@(
  "-u", "scripts/generate_mc_stratified_dynamic_tasksets.py",
  "--candidate-seed-start", "0",
  "--num-candidates", $CandidateCount,
  "--output-manifest", $Manifest,
  "--output-rejections", $Rejections
) + $SchedArgs)

Invoke-PythonStep -Label "D0 static checks" -Arguments @(
  "-u", "scripts/diagnose_mc_stratified_dynamic_structure.py",
  "--manifest", $Manifest,
  "--stage", "D0",
  "--output", $D0
)

Invoke-PythonStep -Label "D1 short characterization" -Arguments @(
  "-u", "scripts/diagnose_mc_stratified_dynamic_structure.py",
  "--manifest", $Manifest,
  "--stage", "D1",
  "--d1-max-candidates", "800",
  "--output", $D1
)

# The D2 diagnostic command applies the declared structure-only cap as the
# downsample boundary; no performance metric is used by the selector.
Invoke-PythonStep -Label "structure-based downsample and D2 diagnostics" -Arguments @(
  "-u", "scripts/diagnose_mc_stratified_dynamic_structure.py",
  "--manifest", $Manifest,
  "--stage", "D2",
  "--d2-max-candidates", "400",
  "--output", $D2
)

Invoke-PythonStep -Label "five-strata selector" -Arguments (@(
  "-u", "scripts/select_mc_stratified_dynamic_primary10.py",
  "--diagnostics", $D2,
  "--output-primary10", $Primary10,
  "--output-audit", $Audit,
  "--output-shortage-report", $Shortage
) + $SchedArgs)

if (-not $DryRun) {
  $Rows = @(Import-Csv -LiteralPath $Primary10)
  $Seeds = @($Rows | ForEach-Object { [int]$_.candidate_seed } | Sort-Object -Unique)
  if ($Rows.Count -ne 10 -or $Seeds.Count -ne 10) {
    throw "primary10 validation failed: rows=$($Rows.Count), unique_candidate_seeds=$($Seeds.Count)"
  }
  Write-Host "primary10 validation: 10 unique candidate seeds"
}
