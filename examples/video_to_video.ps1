# Live2D Anime Generation - Complete Pipeline Script
# Generate Live2D animation from video with all intermediate files

Write-Host "🎬 Starting Live2D Anime Generation Pipeline..." -ForegroundColor Green

# Ensure we're in the correct directory
Set-Location -Path $PSScriptRoot

# Check if input file exists
if (-not (Test-Path "input.mp4")) {
    Write-Host "❌ Error: input.mp4 not found in examples directory" -ForegroundColor Red
    exit 1
}

Write-Host "📁 Input file: input.mp4" -ForegroundColor Cyan
Write-Host "📊 Model: haru_greeter_pro_jp/haru_greeter_t05" -ForegroundColor Cyan
Write-Host "" 

# Run complete video-to-video pipeline
Write-Host "🚀 Running complete pipeline..." -ForegroundColor Yellow

python video_to_video.py `
    --input input.mp4 `
    --output output.mp4 `
    --model live2d_models/haru_greeter_pro_jp/haru_greeter_t05.model3.json `
    --smoothing 0.5 `
    --save-landmarks landmarks.json `
    --save-parameters parameters.json `
    --save-landmark-video landmarks.mp4

# Check generation results
Write-Host ""
Write-Host "📋 Generation Summary:" -ForegroundColor Green

$files = @(
    @{Name="landmarks.json"; Desc="Facial landmark data"},
    @{Name="landmarks.mp4"; Desc="Landmark visualization video"},
    @{Name="parameters.json"; Desc="Live2D parameter data"},
    @{Name="output.mp4"; Desc="Final Live2D animation"}
)

foreach ($file in $files) {
    if (Test-Path $file.Name) {
        $size = (Get-Item $file.Name).Length
        $sizeStr = if ($size -gt 1MB) { "{0:N1} MB" -f ($size/1MB) } else { "{0:N1} KB" -f ($size/1KB) }
        Write-Host "✅ $($file.Name) - $($file.Desc) ($sizeStr)" -ForegroundColor Green
    } else {
        Write-Host "❌ $($file.Name) - Missing!" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🎉 Pipeline completed! You can now:" -ForegroundColor Green
Write-Host "   📺 Watch output.mp4 for the final Live2D animation"
Write-Host "   🔍 Check landmarks.mp4 to verify face detection"
Write-Host "   📊 Analyze landmarks.json and parameters.json for debugging"
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "   • Use different --smoothing values (0.1-0.9) for different smoothness"
Write-Host "   • Replace input.mp4 with your own video file"
Write-Host "   • Check CLAUDE.md for more configuration options"