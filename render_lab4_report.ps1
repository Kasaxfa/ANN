$ErrorActionPreference = 'Stop'
$reportFile = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'lab4_reworked') -Filter '*.docx' | Select-Object -First 1
$wordApp = $null
$reportDoc = $null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $reportDoc = $wordApp.Documents.Open($reportFile.FullName, $false, $false)
    $reportDoc.Fields.Update() | Out-Null
    $reportDoc.Repaginate()
    $reportDoc.Save()
    $pdfPath = [System.IO.Path]::ChangeExtension($reportFile.FullName, '.pdf')
    $reportDoc.ExportAsFixedFormat($pdfPath, 17)
    $pageCount = $reportDoc.ComputeStatistics(2)
    $pages = @()
    foreach ($page in 1..$pageCount) {
        $startRange = $reportDoc.GoTo(1,1,$page)
        $startPosition = $startRange.Start
        if ($page -lt $pageCount) { $endPosition = $reportDoc.GoTo(1,1,($page+1)).Start }
        else { $endPosition = $reportDoc.Content.End }
        $pageRange = $reportDoc.Range($startPosition,$endPosition)
        $text = $pageRange.Text.Trim()
        $pages += [PSCustomObject]@{Page=$page; Characters=$text.Length; Start=$text.Substring(0,[Math]::Min(110,$text.Length)); Pictures=$pageRange.InlineShapes.Count}
    }
    $pages | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportFile.DirectoryName 'document_layout.json') -Encoding UTF8
    Write-Output "Pages: $pageCount"
    $pages | Where-Object {$_.Characters -lt 150} | Format-Table
    Write-Output $pdfPath
} finally {
    if ($null -ne $reportDoc) {$reportDoc.Close(0)}
    if ($null -ne $wordApp) {$wordApp.Quit()}
}
