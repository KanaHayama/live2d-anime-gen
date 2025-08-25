# Live2D Anime Generation - Video Merge Script
# Merge landmarks.mp4 (left) and output.mp4 (right) side by side with NVENC acceleration
#
# Usage:
#   .\merge.ps1                           # Default: 2160x1080 (FHD)
#   .\merge.ps1 -Height 720              # 1440x720 (HD)
#   .\merge.ps1 -Height 1080 -Output my_merged.mp4

param(
    [int]$Height = 1080,  # Default to FHD height, each side will be square (1080x1080)
    [string]$Output = "merged.mp4"
)

Write-Host "🎬 Starting video merge process..." -ForegroundColor Green

# Ensure we're in the correct directory
Set-Location -Path $PSScriptRoot

# Check if required files exist
$landmarksFile = "landmarks.mp4"
$outputFile = "output.mp4"
$mergedFile = $Output

# Calculate target dimensions
$targetHeight = $Height
$targetWidth = $Height  # Each side will be square based on height
$mergedWidth = $targetWidth * 2  # Total width is 2x

if (-not (Test-Path $landmarksFile)) {
    Write-Host "❌ Error: $landmarksFile not found in examples directory" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $outputFile)) {
    Write-Host "❌ Error: $outputFile not found in examples directory" -ForegroundColor Red
    exit 1
}

Write-Host "📁 Input files:" -ForegroundColor Cyan
Write-Host "  Left:  $landmarksFile" -ForegroundColor Cyan
Write-Host "  Right: $outputFile" -ForegroundColor Cyan
Write-Host "  Output: $mergedFile" -ForegroundColor Cyan
Write-Host "📐 Target resolution: ${mergedWidth}x${targetHeight} (${targetWidth}x${targetHeight} each side)" -ForegroundColor Cyan
Write-Host ""

# Get original video properties using ffprobe
Write-Host "🔍 Analyzing original video properties..." -ForegroundColor Yellow
try {
    $landmarksInfo = ffprobe -v quiet -print_format json -show_streams $landmarksFile | ConvertFrom-Json
    $landmarksVideoStream = $landmarksInfo.streams | Where-Object { $_.codec_type -eq "video" } | Select-Object -First 1
    
    $originalWidth = [int]$landmarksVideoStream.width
    $originalHeight = [int]$landmarksVideoStream.height
    $originalFPS = [double]$landmarksVideoStream.r_frame_rate.Split('/')[0] / [double]$landmarksVideoStream.r_frame_rate.Split('/')[1]
    
    Write-Host "  Original resolution: ${originalWidth}x${originalHeight} @ ${originalFPS} fps" -ForegroundColor Green
    Write-Host "  Target resolution: ${mergedWidth}x${targetHeight} @ ${originalFPS} fps" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Failed to analyze video properties" -ForegroundColor Red
    Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Check if NVENC is available
Write-Host "🚀 Checking NVENC availability..." -ForegroundColor Yellow
try {
    $nvencTest = ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=1 -c:v h264_nvenc -f null - 2>&1
    if ($LASTEXITCODE -eq 0) {
        $videoCodec = "h264_nvenc"
        $codecOptions = "-preset fast -cq 23"
        Write-Host "✅ NVENC available - using GPU acceleration" -ForegroundColor Green
    } else {
        throw "NVENC not available"
    }
} catch {
    $videoCodec = "libx264"
    $codecOptions = "-preset fast -crf 23"
    Write-Host "⚠️  NVENC not available - falling back to CPU encoding" -ForegroundColor Yellow
}

Write-Host ""

# Build ffmpeg command
Write-Host "🎥 Starting video merge with ffmpeg..." -ForegroundColor Yellow

$ffmpegArgs = @(
    "-i", $landmarksFile,
    "-i", $outputFile,
    "-filter_complex", "[0:v]scale=${targetWidth}:${targetHeight}[left];[1:v]scale=${targetWidth}:${targetHeight}[right];[left][right]hstack=inputs=2[merged]",
    "-map", "[merged]",
    "-c:v", $videoCodec
)

# Add codec-specific options
$ffmpegArgs += $codecOptions.Split(' ')

# Add output options
$ffmpegArgs += @(
    "-r", $originalFPS.ToString(),
    "-y",  # Overwrite output file
    $mergedFile
)

Write-Host "Command: ffmpeg $($ffmpegArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

try {
    # Run ffmpeg
    & ffmpeg $ffmpegArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Video merge completed successfully!" -ForegroundColor Green
        
        # Show output file info
        if (Test-Path $mergedFile) {
            $mergedSize = (Get-Item $mergedFile).Length
            $mergedSizeStr = if ($mergedSize -gt 1MB) { "{0:N1} MB" -f ($mergedSize/1MB) } else { "{0:N1} KB" -f ($mergedSize/1KB) }
            Write-Host "📄 Output: $mergedFile ($mergedSizeStr)" -ForegroundColor Green
            
            # Verify merged video properties
            Write-Host ""
            Write-Host "🔍 Verifying merged video..." -ForegroundColor Yellow
            try {
                $mergedInfo = ffprobe -v quiet -print_format json -show_streams $mergedFile | ConvertFrom-Json
                $mergedVideoStream = $mergedInfo.streams | Where-Object { $_.codec_type -eq "video" } | Select-Object -First 1
                
                $actualWidth = [int]$mergedVideoStream.width
                $actualHeight = [int]$mergedVideoStream.height
                $actualFPS = [double]$mergedVideoStream.r_frame_rate.Split('/')[0] / [double]$mergedVideoStream.r_frame_rate.Split('/')[1]
                
                Write-Host "✅ Merged video: ${actualWidth}x${actualHeight} @ ${actualFPS} fps" -ForegroundColor Green
                
                if ($actualWidth -eq $mergedWidth -and $actualHeight -eq $targetHeight) {
                    Write-Host "✅ Dimensions verified correctly" -ForegroundColor Green
                } else {
                    Write-Host "⚠️  Dimension mismatch - expected ${mergedWidth}x${targetHeight}" -ForegroundColor Yellow
                }
            } catch {
                Write-Host "⚠️  Could not verify merged video properties" -ForegroundColor Yellow
            }
        }
    } else {
        throw "FFmpeg failed with exit code $LASTEXITCODE"
    }
} catch {
    Write-Host ""
    Write-Host "❌ Error during video merge: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 Merge process completed!" -ForegroundColor Green
Write-Host "💡 You can now watch $mergedFile to see landmarks (left) and Live2D animation (right) side by side" -ForegroundColor Yellow
Write-Host ""
Write-Host "📊 File Summary:" -ForegroundColor Green
$files = @(
    @{Name=$landmarksFile; Desc="Landmarks visualization"},
    @{Name=$outputFile; Desc="Live2D animation"},
    @{Name=$mergedFile; Desc="Side-by-side comparison"}
)

foreach ($file in $files) {
    if (Test-Path $file.Name) {
        $size = (Get-Item $file.Name).Length
        $sizeStr = if ($size -gt 1MB) { "{0:N1} MB" -f ($size/1MB) } else { "{0:N1} KB" -f ($size/1KB) }
        Write-Host "  📄 $($file.Name) - $($file.Desc) ($sizeStr)" -ForegroundColor Green
    }
}