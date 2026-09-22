$ErrorActionPreference = 'Stop'
$bookPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'lab4_reworked\water_variant17_lab4_work.xlsx')).Path
$excelApp = $null
$labBook = $null
try {
    $excelApp = New-Object -ComObject Excel.Application
    $excelApp.Visible = $false
    $excelApp.DisplayAlerts = $false
    $excelApp.AskToUpdateLinks = $false
    $labBook = $excelApp.Workbooks.Open($bookPath, 0, $false)
    $excelApp.CalculateFullRebuild()
    $labBook.Save()
    $summarySheet = $labBook.Worksheets.Item('Summary')
    foreach ($column in 2..5) {
        [PSCustomObject]@{
            Model = $summarySheet.Cells.Item(1,$column).Text
            Errors = $summarySheet.Cells.Item(2,$column).Value2
            Accuracy = $summarySheet.Cells.Item(3,$column).Value2
            Mismatches = $summarySheet.Cells.Item(4,$column).Value2
            MaxLogitDifference = $summarySheet.Cells.Item(5,$column).Value2
        } | Format-List
        if ($summarySheet.Cells.Item(4,$column).Value2 -ne 0) { throw 'Excel predictions do not match Python.' }
    }
} finally {
    if ($null -ne $labBook) { $labBook.Close($false) }
    if ($null -ne $excelApp) { $excelApp.Quit() }
}
