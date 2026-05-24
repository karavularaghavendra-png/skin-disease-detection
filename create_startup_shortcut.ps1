# Create a Windows Startup shortcut for auto_start.bat
$WshShell = New-Object -ComObject WScript.Shell
$StartupFolder = [System.IO.Path]::Combine($env:APPDATA, 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup')
$ShortcutPath = [System.IO.Path]::Combine($StartupFolder, 'SkinDiseaseDetection.lnk')

$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = 'C:\Users\karav\OneDrive\Desktop\skin_disease_detection\auto_start.bat'
$Shortcut.WorkingDirectory = 'C:\Users\karav\OneDrive\Desktop\skin_disease_detection'
$Shortcut.WindowStyle = 7  # Minimized
$Shortcut.Description = 'Auto-start Skin Disease Detection server and ngrok'
$Shortcut.Save()

Write-Host "Startup shortcut created at: $ShortcutPath"
Write-Host "The server and ngrok will now auto-start every time you log in."
